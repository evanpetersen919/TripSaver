import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

export const metadata: Metadata = {
  title: "TripSaver - AI-Powered Travel Planning",
  description: "Transform your travel photos into perfectly optimized itineraries with AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {/* Header */}
        <header 
          className="fixed top-0 left-0 right-0 z-50 bg-zinc-900 bg-opacity-10 backdrop-blur-md px-8 py-6 border-b border-stone-800 border-opacity-30"
          style={{ 
            backdropFilter: 'blur(16px) saturate(180%)', 
            WebkitBackdropFilter: 'blur(16px) saturate(180%)',
            backgroundColor: 'rgba(24, 24, 27, 0.1)'
          } as React.CSSProperties}
        >
          <div className="max-w-7xl mx-auto flex justify-between items-center">
            <Link href="/" className="text-2xl font-light text-orange-400 hover:text-orange-300 transition-colors">TripSaver</Link>
            <div className="flex gap-4">
              <Link href="/login" className="px-4 py-2 text-orange-400 hover:text-orange-300 font-light transition-colors">
                Log In
              </Link>
              <Link href="/signup" className="px-4 py-2 bg-orange-500 bg-opacity-20 text-white rounded-lg hover:bg-opacity-30 font-light transition-all border border-orange-500 border-opacity-30">
                Sign Up
              </Link>
            </div>
          </div>
        </header>
        <main className="pt-0">
          {children}
        </main>
      </body>
    </html>
  );
}
