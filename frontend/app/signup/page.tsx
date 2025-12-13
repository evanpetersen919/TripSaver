'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api';

export default function SignUp() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (username.length < 3) {
      setError('Username must be at least 3 characters');
      return;
    }

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
      await apiClient.signup(email, password, username);
      router.push('/plan');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create account. Please try again.');
      setLoading(false);
    }
  };

  const handleGoogleSignIn = () => {
    // TODO: Implement Google OAuth
    console.log('Google sign in clicked');
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-900 via-stone-900 to-zinc-900 flex items-center justify-center px-8 py-32">
      <style jsx>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-10px); }
          75% { transform: translateX(10px); }
        }
        .animate-fade-in {
          animation: fadeIn 0.5s ease-out;
        }
      `}</style>
      <div 
        className="w-full max-w-md bg-zinc-900 bg-opacity-70 backdrop-blur-md rounded-3xl p-12 border border-stone-700 border-opacity-30"
        style={{ 
          backdropFilter: 'blur(20px) saturate(180%)', 
          WebkitBackdropFilter: 'blur(20px) saturate(180%)',
          backgroundColor: 'rgba(24, 24, 27, 0.7)',
          animation: 'fadeInUp 0.6s ease-out'
        } as React.CSSProperties}
      >
        <h1 className="text-4xl font-semibold text-stone-100 mb-2 text-center animate-fade-in" style={{ animationDelay: '0.1s', animationFillMode: 'backwards' } as React.CSSProperties}>Create Account</h1>
        <p className="text-stone-400 text-center mb-8 animate-fade-in" style={{ animationDelay: '0.2s', animationFillMode: 'backwards' } as React.CSSProperties}>Start planning your trips with TripSaver</p>

        <button
          onClick={handleGoogleSignIn}
          className="w-full bg-white text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-100 hover:scale-[1.02] hover:shadow-lg active:scale-[0.98] transition-all duration-200 border border-gray-300 font-bold flex items-center justify-center gap-3 mb-6 animate-fade-in"
          style={{ animationDelay: '0.3s', animationFillMode: 'backwards' } as React.CSSProperties}
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
          <div className="bg-red-500 bg-opacity-10 border border-red-500 border-opacity-30 text-red-400 px-4 py-3 rounded-lg mb-6"
            style={{ animation: 'shake 0.4s ease-in-out, fadeIn 0.3s ease-out' } as React.CSSProperties}
          >
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
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-40 rounded-lg text-stone-200 placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-60 focus:ring-2 focus:ring-orange-400 focus:ring-opacity-20 focus:scale-[1.01] transition-all duration-200"
              placeholder="Email address"
            />
          </div>

          <div>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              minLength={3}
              maxLength={30}
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-40 rounded-lg text-stone-200 placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-60 focus:ring-2 focus:ring-orange-400 focus:ring-opacity-20 focus:scale-[1.01] transition-all duration-200"
              placeholder="Username"
            />
          </div>

          <div>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-40 rounded-lg text-stone-200 placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-60 focus:ring-2 focus:ring-orange-400 focus:ring-opacity-20 focus:scale-[1.01] transition-all duration-200"
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
              className="w-full px-4 py-3 bg-zinc-800 bg-opacity-50 border border-stone-700 border-opacity-40 rounded-lg text-stone-200 placeholder-stone-500 focus:outline-none focus:border-orange-400 focus:border-opacity-60 focus:ring-2 focus:ring-orange-400 focus:ring-opacity-20 focus:scale-[1.01] transition-all duration-200"
              placeholder="Confirm password"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-orange-500 bg-opacity-20 text-white px-6 py-3 rounded-lg hover:bg-opacity-30 hover:scale-[1.02] hover:shadow-lg active:scale-[0.98] transition-all duration-200 border border-orange-500 border-opacity-30 font-normal disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Creating Account...
              </span>
            ) : 'Create Account'}
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
