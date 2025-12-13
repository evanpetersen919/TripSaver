import { NextRequest, NextResponse } from 'next/server';

const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID!;
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET!;
const REDIRECT_URI = process.env.NEXT_PUBLIC_GOOGLE_REDIRECT_URI || 
  (process.env.VERCEL_URL 
    ? `https://${process.env.VERCEL_URL}/api/auth/google/callback`
    : 'http://localhost:3000/api/auth/google/callback');
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://eh5scbzco7.execute-api.us-east-1.amazonaws.com/prod';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const code = searchParams.get('code');
  const state = searchParams.get('state'); // 'login' or 'signup'
  const error = searchParams.get('error');

  if (error) {
    return NextResponse.redirect(
      new URL(`/login?error=${encodeURIComponent('Google authentication failed')}`, request.url)
    );
  }

  if (!code) {
    return NextResponse.redirect(
      new URL('/login?error=no_code', request.url)
    );
  }

  try {
    console.log('[OAuth] Starting token exchange with code:', code?.substring(0, 10) + '...');
    console.log('[OAuth] Redirect URI:', REDIRECT_URI);
    
    // Exchange code for tokens
    const tokenResponse = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        code,
        client_id: GOOGLE_CLIENT_ID,
        client_secret: GOOGLE_CLIENT_SECRET,
        redirect_uri: REDIRECT_URI,
        grant_type: 'authorization_code',
      }),
    });

    if (!tokenResponse.ok) {
      const errorData = await tokenResponse.json();
      console.error('[OAuth] Token exchange failed:', errorData);
      throw new Error(`Failed to exchange code for token: ${JSON.stringify(errorData)}`);
    }

    const tokens = await tokenResponse.json();
    console.log('[OAuth] Token exchange successful');

    // Get user info from Google
    const userInfoResponse = await fetch('https://www.googleapis.com/oauth2/v2/userinfo', {
      headers: { Authorization: `Bearer ${tokens.access_token}` },
    });

    if (!userInfoResponse.ok) {
      throw new Error('Failed to get user info');
    }

    const userInfo = await userInfoResponse.json();
    console.log('[OAuth] Got user info:', userInfo.email);

    // Call backend Google auth endpoint
    console.log('[OAuth] Calling backend:', `${API_BASE_URL}/auth/google`);
    const backendResponse = await fetch(`${API_BASE_URL}/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: userInfo.email,
        google_id: userInfo.id,
        name: userInfo.name,
        picture: userInfo.picture,
      }),
    });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error('[OAuth] Backend auth failed:', errorText);
      throw new Error(`Backend authentication failed: ${errorText}`);
    }

    const authResult = await backendResponse.json();
    console.log('[OAuth] Backend auth successful, redirecting to /plan');

    // Redirect to plan page with token in URL (frontend will store it)
    const redirectUrl = new URL('/plan', request.url);
    redirectUrl.searchParams.append('token', authResult.access_token);
    redirectUrl.searchParams.append('username', authResult.username || 'user');
    
    return NextResponse.redirect(redirectUrl.toString());

  } catch (error) {
    console.error('[OAuth] Google OAuth error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Authentication failed';
    return NextResponse.redirect(
      new URL(`/login?error=${encodeURIComponent(errorMessage)}`, request.url)
    );
  }
}
