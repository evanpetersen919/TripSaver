'use client';

import { useState } from 'react';
import Link from 'next/link';

export default function SignUp() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setLoading(true);

    try {
      // TODO: Implement actual signup API call
      console.log('Sign up:', { email, password });
      // Placeholder for API integration
      // const response = await fetch('/api/signup', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ email, password })
      // });
      
      setLoading(false);
    } catch (err) {
      setError('Failed to create account. Please try again.');
      setLoading(false);
    }
  };

  const handleGoogleSignIn = () => {
    // TODO: Implement Google OAuth
    console.log('Google sign in clicked');
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-900 via-stone-900 to-zinc-900 flex items-center justify-center px-8 py-32">
      <div 
        className="w-full max-w-md bg-zinc-900 bg-opacity-70 backdrop-blur-md rounded-3xl p-12 border border-stone-700 border-opacity-30"
        style={{ 
          backdropFilter: 'blur(20px) saturate(180%)', 
          WebkitBackdropFilter: 'blur(20px) saturate(180%)',
          backgroundColor: 'rgba(24, 24, 27, 0.7)'
        } as React.CSSProperties}
      >
        <h1 className="text-4xl font-semibold text-stone-100 mb-2 text-center">Create Account</h1>
        <p className="text-stone-400 text-center mb-8">Start planning your trips with TripSaver</p>

        <button
          onClick={handleGoogleSignIn}
          className="w-full bg-white text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-100 transition-all border border-gray-300 font-bold flex items-center justify-center gap-3 mb-6"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
          </svg>
          Sign up with Google
        </button>

        <div className="relative flex items-center justify-center mb-6">
          <div className="border-t border-stone-700 border-opacity-40 w-full"></div>
          <span className="absolute bg-zinc-900 bg-opacity-70 px-4 text-stone-500 text-sm">or</span>
        </div>

        {error && (
          <div className="bg-red-500 bg-opacity-10 border border-red-500 border-opacity-30 text-red-400 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-40 rounded-lg text-stone-200 placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-60 transition-colors"
              placeholder="Email address"
            />
          </div>

          <div>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-40 rounded-lg text-stone-200 placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-60 transition-colors"
              placeholder="Password"
            />
          </div>

          <div>
            <input
              type="password"
              id="confirmPassword"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-40 rounded-lg text-stone-200 placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-60 transition-colors"
              placeholder="Confirm password"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-orange-500 bg-opacity-20 text-white px-6 py-3 rounded-lg hover:bg-opacity-30 transition-all border border-orange-500 border-opacity-30 font-normal disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        <p className="text-stone-400 text-center mt-8 text-sm">
          Already have an account?{' '}
          <Link href="/login" className="text-orange-400 hover:text-orange-300 transition-colors font-normal">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
