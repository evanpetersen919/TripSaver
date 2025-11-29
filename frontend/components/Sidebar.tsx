'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  HomeIcon, 
  MapIcon, 
  UserIcon, 
  CloudArrowUpIcon,
  ArrowRightOnRectangleIcon 
} from '@heroicons/react/24/outline';
import { useEffect, useState } from 'react';

export default function Sidebar() {
  const pathname = usePathname();
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    setIsLoggedIn(!!localStorage.getItem('token'));
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsLoggedIn(false);
    window.location.href = '/';
  };

  const navigation = [
    { name: 'Home', href: '/', icon: HomeIcon },
    { name: 'Upload Image', href: '/upload', icon: CloudArrowUpIcon },
    { name: 'My Itinerary', href: '/itinerary', icon: MapIcon, requiresAuth: true },
    { name: 'Profile', href: '/profile', icon: UserIcon, requiresAuth: true },
  ];

  return (
    <div className="flex flex-col w-64 bg-white border-r border-gray-200">
      {/* Logo */}
      <div className="flex items-center h-16 px-6 border-b border-gray-200">
        <h1 className="text-xl font-bold text-blue-600">TripSaver</h1>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {navigation.map((item) => {
          if (item.requiresAuth && !isLoggedIn) return null;
          
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                isActive
                  ? 'bg-primary-50 text-primary-700'
                  : 'text-gray-700 hover:bg-gray-50'
              }`}
            >
              <item.icon className="w-5 h-5 mr-3" />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Auth Section */}
      <div className="p-4 border-t border-gray-200">
        {isLoggedIn ? (
          <button
            onClick={handleLogout}
            className="flex items-center w-full px-4 py-3 text-sm font-medium text-gray-700 rounded-lg hover:bg-gray-50"
          >
            <ArrowRightOnRectangleIcon className="w-5 h-5 mr-3" />
            Logout
          </button>
        ) : (
          <Link
            href="/login"
            className="flex items-center justify-center w-full px-4 py-3 text-sm font-medium text-white bg-primary-600 rounded-lg hover:bg-primary-700"
          >
            Login / Sign Up
          </Link>
        )}
      </div>
    </div>
  );
}
