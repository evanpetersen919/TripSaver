'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useState } from 'react';

export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [archRotation, setArchRotation] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(0);
  const [currentRotation, setCurrentRotation] = useState(0);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart(e.clientX);
    setCurrentRotation(archRotation);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const delta = e.clientX - dragStart;
    const rotation = currentRotation + delta * 0.3; // 0.3 is sensitivity
    setArchRotation(rotation);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header/Navigation */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-zinc-900/95 backdrop-blur-md border-b border-stone-800">
        <nav className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link href="/" className="flex items-center space-x-2 group">
              <Image 
                src="/images/logo_text.png" 
                alt="TripSaver" 
                width={150} 
                height={40} 
                className="h-8 w-auto group-hover:scale-105 transition-transform duration-200"
              />
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-stone-300 hover:text-orange-400 transition-colors duration-200 font-medium">Features</a>
              <a href="#how-it-works" className="text-stone-300 hover:text-orange-400 transition-colors duration-200 font-medium">How It Works</a>
              <a href="#about" className="text-stone-300 hover:text-orange-400 transition-colors duration-200 font-medium">About</a>
              <Link 
                href="/login" 
                className="px-6 py-2 bg-gradient-to-r from-orange-500 to-red-600 text-white rounded-lg hover:shadow-lg hover:shadow-orange-500/50 transition-all duration-200 font-semibold"
              >
                Get Started
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <button 
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden text-stone-300 hover:text-orange-400 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {mobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden mt-4 pb-4 space-y-3 border-t border-stone-800 pt-4">
              <a href="#features" className="block text-stone-300 hover:text-orange-400 transition-colors font-medium">Features</a>
              <a href="#how-it-works" className="block text-stone-300 hover:text-orange-400 transition-colors font-medium">How It Works</a>
              <a href="#about" className="block text-stone-300 hover:text-orange-400 transition-colors font-medium">About</a>
              <Link 
                href="/login" 
                className="block text-center px-6 py-2 bg-gradient-to-r from-orange-500 to-red-600 text-white rounded-lg font-semibold"
              >
                Get Started
              </Link>
            </div>
          )}
        </nav>
      </header>

      {/* Hero Section with Background Image */}
      <div className="relative pt-48 pb-32 px-8 min-h-screen bg-gradient-to-br from-orange-950 via-zinc-900 to-stone-900">
        {/* Background Image - Japan */}
        <div 
          className="absolute inset-0 z-0 opacity-80"
          style={{ 
            backgroundImage: 'url(/images/hero-bg.jpg)',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            filter: 'brightness(0.6)'
          }}
        />
        
        {/* Dark overlay */}
        <div className="absolute inset-0 bg-black/20 z-0"></div>
        
        {/* Content */}
        <div className="relative z-10 max-w-5xl mx-auto text-center">
          <div className="flex justify-center mb-4">
            <Image src="/images/logo_text.png" alt="TripSaver" width={700} height={400} quality={100} className="h-auto hover:scale-105 transition-transform duration-300 cursor-pointer" />
          </div>
          <p className="text-xl text-stone-200 max-w-2xl mx-auto mb-10 font-light leading-relaxed">
            Transform your travel photos into landmark discoveries with AI-powered visual recognition and intelligent location recommendations
          </p>
          <Link
            href="/login"
            className="group relative inline-block overflow-hidden border-2 border-orange-500 text-orange-400 hover:text-white px-12 py-5 rounded-2xl text-lg font-semibold hover:scale-105 hover:shadow-[0_0_40px_rgba(249,115,22,0.6)] transition-all duration-300 mb-16"
          >
            <span className="relative z-10 flex items-center gap-2">
              Start Planning Now
              <svg className="w-5 h-5 group-hover:translate-x-2 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </span>
            <div className="absolute inset-0 bg-gradient-to-r from-orange-500 via-red-600 to-amber-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          </Link>
          
          {/* Demo Video Placeholder */}
          <div className="relative rounded-2xl shadow-2xl aspect-video max-w-5xl mx-auto overflow-hidden border border-stone-700 border-opacity-50 bg-black bg-opacity-30 backdrop-blur-sm hover:scale-105 hover:shadow-[0_0_50px_rgba(249,115,22,0.4)] transition-all duration-500 group cursor-pointer">
            <div className="absolute inset-0 flex items-center justify-center z-10">
              <div className="text-center">
                <div className="text-6xl text-orange-400 drop-shadow-lg mb-4 group-hover:scale-125 group-hover:text-orange-300 transition-all duration-300">▶</div>
                <p className="text-white text-lg font-light drop-shadow-lg group-hover:text-orange-300 transition-colors duration-300">Demo Video Coming Soon</p>
              </div>
            </div>
          </div>

          {/* Features Section */}
          <div id="features" className="relative z-10 mt-48">
            <h2 className="text-4xl font-bold text-center text-stone-100 mb-16">Key Features</h2>
            <div className="space-y-8 max-w-6xl mx-auto">
              {/* Feature 1: Upload & Identify */}
              <div className="relative overflow-hidden rounded-3xl border border-stone-700 border-opacity-30 group hover:border-orange-400 hover:border-opacity-50 transition-all duration-500 hover:shadow-[0_0_50px_rgba(251,146,60,0.3)]">
                <div className="flex flex-col md:flex-row">
                  <div className="md:w-1/2 relative h-80 md:h-auto overflow-hidden">
                    <Image
                      src="https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&h=600&fit=crop"
                      alt="Eiffel Tower Paris"
                      width={800}
                      height={600}
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                    />
                  </div>
                  <div className="md:w-1/2 p-10 bg-zinc-900 bg-opacity-90 backdrop-blur-md flex flex-col justify-center">
                    <h3 className="text-3xl font-bold text-stone-100 mb-4 group-hover:text-orange-400 transition-colors duration-300">Upload & Identify</h3>
                    <p className="text-stone-300 text-lg leading-relaxed">
                      Instantly recognize landmarks from any travel photo or screenshot.
                    </p>
                  </div>
                </div>
              </div>

              {/* Feature 2: Visual Similarity Search */}
              <div className="relative overflow-hidden rounded-3xl border border-stone-700 border-opacity-30 group hover:border-orange-400 hover:border-opacity-50 transition-all duration-500 hover:shadow-[0_0_50px_rgba(251,146,60,0.3)]">
                <div className="flex flex-col md:flex-row-reverse">
                  <div className="md:w-1/2 relative h-80 md:h-auto overflow-hidden">
                    <Image
                      src="https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=800&h=600&fit=crop"
                      alt="Tokyo Tower Japan"
                      width={800}
                      height={600}
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                    />
                  </div>
                  <div className="md:w-1/2 p-10 bg-zinc-900 bg-opacity-90 backdrop-blur-md flex flex-col justify-center">
                    <h3 className="text-3xl font-bold text-stone-100 mb-4 group-hover:text-orange-400 transition-colors duration-300">Visual Similarity Search</h3>
                    <p className="text-stone-300 text-lg leading-relaxed">
                      Find visually similar landmarks across a database of 4200+ locations.
                    </p>
                  </div>
                </div>
              </div>

              {/* Feature 3: Smart Recommendations */}
              <div className="relative overflow-hidden rounded-3xl border border-stone-700 border-opacity-30 group hover:border-orange-400 hover:border-opacity-50 transition-all duration-500 hover:shadow-[0_0_50px_rgba(251,146,60,0.3)]">
                <div className="flex flex-col md:flex-row">
                  <div className="md:w-1/2 relative h-80 md:h-auto overflow-hidden">
                    <Image
                      src="https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=800&h=600&fit=crop"
                      alt="Colosseum Rome"
                      width={800}
                      height={600}
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                    />
                  </div>
                  <div className="md:w-1/2 p-10 bg-zinc-900 bg-opacity-90 backdrop-blur-md flex flex-col justify-center">
                    <h3 className="text-3xl font-bold text-stone-100 mb-4 group-hover:text-orange-400 transition-colors duration-300">Smart Recommendations</h3>
                    <p className="text-stone-300 text-lg leading-relaxed">
                      Receive nearby landmark suggestions tailored to your detected locations.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Blocks - Vertically Stacked */}
      <div id="how-it-works" className="py-24 px-8 bg-gradient-to-b from-stone-900 via-zinc-900 to-stone-900">
        <h2 className="text-5xl font-bold text-center text-stone-100 mb-24 tracking-tight">Technical Deep Dive</h2>
        <div className="max-w-4xl mx-auto space-y-56">
          {/* Training */}
          <div className="border-l-2 border-orange-400 border-opacity-40 pl-12 hover:border-opacity-100 hover:pl-14 hover:-mr-2 transition-all duration-300 group">
            <div className="min-w-0">
            <h2 className="text-5xl font-semibold text-stone-100 mb-6 tracking-tight group-hover:text-orange-400 transition-colors duration-300">Training</h2>
            <p className="text-stone-200 text-lg leading-relaxed mb-8 font-normal">
              EfficientNet-B3 was trained on RTX 4080 using Google Landmarks Dataset v2, starting with ImageNet pretrained weights and fine-tuning on 500 landmark classes. Training employed standard augmentations including RandAugment, MixUp, and CutMix, plus custom social media augmentation to improve robustness on screenshot inputs.
            </p>
            <ul className="space-y-5">
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Mixed precision FP16 training with AdamW optimizer and cosine annealing schedule. Gradient accumulation over 4 steps and early stopping prevent overfitting on imbalanced classes</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Custom augmentation synthesizes Instagram/TikTok UI overlays (profile bars, story circles, filters) at 30% probability, enabling accurate predictions on social media screenshots</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Dataset filtered from 200K+ landmarks to 500 classes with sufficient training samples and global geographic diversity for practical travel applications</span>
              </li>
            </ul>
            </div>
          </div>

          {/* How It Works */}
          <div className="border-l-2 border-orange-400 border-opacity-40 pl-12 hover:border-opacity-100 hover:pl-14 hover:-mr-2 transition-all duration-300 group">
            <div className="min-w-0">
            <h2 className="text-5xl font-semibold text-stone-100 mb-6 tracking-tight group-hover:text-orange-400 transition-colors duration-300">How It Works</h2>
            
            <p className="text-stone-200 text-lg leading-relaxed mb-8 font-normal">
              The system uses a two-tier detection approach. Tier 1 runs EfficientNet-B3 with optional Google Vision API validation for low-confidence results. Tier 2 is triggered when users reject initial predictions, using CLIP embeddings for visual similarity and Groq's vision-language model for scene understanding.
            </p>
            <ul className="space-y-5">
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Uploaded images are preprocessed to remove social media UI elements using template matching. Images are resized to 300x300 and normalized with ImageNet statistics before inference</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Tier 1 attempts fast classification with EfficientNet-B3 (500 classes). If confidence is below 70%, Google Vision API validates the prediction. Results are stored in DynamoDB for potential fallback</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Tier 2 activates when users reject initial results. CLIP ViT-B/32 generates embeddings for FAISS similarity search across 4200+ landmarks. Groq's Llama-4-Scout-17B provides detailed scene descriptions and context</span>
              </li>
            </ul>
            
            {/* Detection Flow Diagram */}
            <div className="mt-10">
              {/* Tier 1 */}
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-orange-400 mb-4 text-center">Tier 1: Primary Detection</h3>
                <div className="flex flex-col md:flex-row items-center justify-center gap-4">
                  <div className="bg-zinc-800/50 p-4 rounded-lg border border-orange-400/30 text-center min-w-[140px]">
                    <Image src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="EfficientNet-B3" width={40} height={40} className="mx-auto mb-2"/>
                    <p className="text-stone-200 text-sm font-semibold">EfficientNet-B3</p>
                    <p className="text-stone-500 text-xs">Trained Model</p>
                  </div>

                  <div className="flex flex-col items-center gap-2">
                    <svg className="w-6 h-6 text-orange-400 rotate-90 md:rotate-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                    <span className="text-xs text-stone-400">if confidence &lt; 70%</span>
                    <svg className="w-6 h-6 text-orange-400 rotate-90 md:rotate-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </div>

                  <div className="bg-zinc-800/50 p-4 rounded-lg border border-orange-400/30 text-center min-w-[140px]">
                    <Image src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" alt="Google Vision" width={40} height={40} className="mx-auto mb-2"/>
                    <p className="text-stone-200 text-sm font-semibold">Vision API</p>
                    <p className="text-stone-500 text-xs">Validation</p>
                  </div>
                </div>
              </div>

              {/* Separator */}
              <div className="relative my-8">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t-2 border-stone-700"></div>
                </div>
                <div className="relative flex justify-center">
                  <span className="bg-zinc-900 px-4 text-stone-400 text-sm">User rejects prediction</span>
                </div>
              </div>

              {/* Tier 2 */}
              <div className="mt-6">
                <h3 className="text-xl font-semibold text-orange-400 mb-4 text-center">Tier 2: Fallback Analysis</h3>
                <div className="flex flex-col md:flex-row items-center justify-center gap-4">
                  <div className="bg-zinc-800/50 p-4 rounded-lg border border-orange-400/30 text-center min-w-[140px]">
                    <svg className="w-10 h-10 mx-auto mb-2" viewBox="0 0 24 24" fill="none">
                      <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z" fill="#10A37F"/>
                    </svg>
                    <p className="text-stone-200 text-sm font-semibold">CLIP ViT-B/32</p>
                    <p className="text-stone-500 text-xs">Similarity Search</p>
                  </div>

                  <span className="text-stone-400 text-lg">+</span>

                  <div className="bg-zinc-800/50 p-4 rounded-lg border border-orange-400/30 text-center min-w-[140px]">
                    <div className="w-10 h-10 mx-auto mb-2 flex items-center justify-center bg-white rounded-lg">
                      <span className="text-xl font-bold text-black">G</span>
                    </div>
                    <p className="text-stone-200 text-sm font-semibold">Groq LLM</p>
                    <p className="text-stone-500 text-xs">Scene Description</p>
                  </div>
                </div>
              </div>
            </div>
            </div>
          </div>

          {/* System Architecture */}
          <div className="border-l-2 border-orange-400 border-opacity-40 pl-12 hover:border-opacity-100 hover:pl-14 hover:-mr-2 transition-all duration-300 group">
            <div className="min-w-0">
            <h2 className="text-5xl font-semibold text-stone-100 mb-6 tracking-tight group-hover:text-orange-400 transition-colors duration-300">System Architecture</h2>
            <p className="text-stone-200 text-lg font-normal leading-relaxed mb-8">
              The frontend is deployed on Vercel, routing requests through AWS API Gateway to Lambda functions. FastAPI handles authentication and routing, calling the EfficientNet-B3 model on HuggingFace Spaces. User data and itineraries are stored in DynamoDB with Google Cloud Vision API and Groq providing fallback detection.
            </p>
            <ul className="space-y-5">
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Next.js frontend deploys to Vercel with automatic builds on GitHub pushes, ensuring fast global delivery via edge network with sub-100ms response times</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">AWS API Gateway routes requests to Lambda functions running FastAPI with JWT authentication. DynamoDB stores user profiles and itineraries with GSI indexes for efficient queries</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">EfficientNet-B3 model runs on HuggingFace Spaces with CPU. CloudWatch monitors Lambda performance while GitHub Actions automates testing and deployment workflows</span>
              </li>
            </ul>
            
            {/* 3D Carousel Architecture Diagram */}
            <div className="mt-10 relative">
              <div className="flex items-center justify-center mb-4">
                <div className="text-stone-400 text-sm font-medium flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                  </svg>
                  Drag to explore architecture
                </div>
              </div>

              <div 
                className="relative h-[500px] perspective-[1200px] cursor-grab active:cursor-grabbing select-none"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              >
                <div
                  className={`relative w-full h-full ${isDragging ? '' : 'transition-transform duration-300 ease-out'}`}
                  style={{
                    transformStyle: 'preserve-3d',
                    transform: `rotateY(${archRotation}deg)`,
                    willChange: 'transform',
                  }}
                >
                  {/* Frontend - Position 0 */}
                  <div
                    className="absolute top-1/2 left-1/2 w-64"
                    style={{
                      transform: 'translate(-50%, -50%) rotateY(0deg) translateZ(280px)',
                      transformStyle: 'preserve-3d',
                    }}
                  >
                    <div style={{ transform: `rotateY(${-archRotation}deg)` }}>
                      <h3 className="text-sm font-semibold text-orange-400 mb-3 text-center">Frontend</h3>
                      <div className="bg-zinc-800 rounded-xl p-4 border border-stone-700 space-y-3">
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <Image src="https://www.vectorlogo.zone/logos/nextjs/nextjs-icon.svg" alt="Next.js" width={28} height={28} className="invert mx-auto mb-1"/>
                          <p className="text-stone-200 text-xs font-semibold">Next.js 16</p>
                          <p className="text-stone-500 text-xs">React 19</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <svg className="w-7 h-7 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                            <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z" fill="#06B6D4"/>
                          </svg>
                          <p className="text-stone-200 text-xs font-semibold">Tailwind</p>
                          <p className="text-stone-500 text-xs">Styling</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <svg className="w-7 h-7 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="#10B981"/>
                          </svg>
                          <p className="text-stone-200 text-xs font-semibold">Vercel</p>
                          <p className="text-stone-500 text-xs">Hosting</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Backend - Position 60deg */}
                  <div
                    className="absolute top-1/2 left-1/2 w-64"
                    style={{
                      transform: 'translate(-50%, -50%) rotateY(60deg) translateZ(280px)',
                      transformStyle: 'preserve-3d',
                    }}
                  >
                    <div style={{ transform: `rotateY(${-archRotation - 60}deg)` }}>
                      <h3 className="text-sm font-semibold text-orange-400 mb-3 text-center">Backend</h3>
                      <div className="bg-zinc-800 rounded-xl p-4 border border-stone-700 space-y-3">
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <Image src="https://www.vectorlogo.zone/logos/amazon_aws/amazon_aws-icon.svg" alt="API Gateway" width={28} height={28} className="mx-auto mb-1"/>
                          <p className="text-stone-200 text-xs font-semibold">API Gateway</p>
                          <p className="text-stone-500 text-xs">REST</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <svg className="w-7 h-7 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                            <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" fill="#009688"/>
                          </svg>
                          <p className="text-stone-200 text-xs font-semibold">FastAPI</p>
                          <p className="text-stone-500 text-xs">Lambda</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <svg className="w-7 h-7 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                            <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z" fill="#F59E0B"/>
                          </svg>
                          <p className="text-stone-200 text-xs font-semibold">JWT Auth</p>
                          <p className="text-stone-500 text-xs">Security</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* ML Inference - Position 120deg */}
                  <div
                    className="absolute top-1/2 left-1/2 w-64"
                    style={{
                      transform: 'translate(-50%, -50%) rotateY(120deg) translateZ(280px)',
                      transformStyle: 'preserve-3d',
                    }}
                  >
                    <div style={{ transform: `rotateY(${-archRotation - 120}deg)` }}>
                      <h3 className="text-sm font-semibold text-orange-400 mb-3 text-center">ML Inference</h3>
                      <div className="bg-zinc-800 rounded-xl p-4 border border-stone-700 space-y-3">
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <Image src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="PyTorch" width={28} height={28} className="mx-auto mb-1"/>
                          <p className="text-stone-200 text-xs font-semibold">EfficientNet-B3</p>
                          <p className="text-stone-500 text-xs">HF Spaces</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <Image src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" alt="Google Vision" width={28} height={28} className="mx-auto mb-1"/>
                          <p className="text-stone-200 text-xs font-semibold">Vision API</p>
                          <p className="text-stone-500 text-xs">Fallback</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <div className="w-7 h-7 mx-auto mb-1 flex items-center justify-center bg-white rounded-lg">
                            <span className="text-sm font-bold text-black">G</span>
                          </div>
                          <p className="text-stone-200 text-xs font-semibold">Groq</p>
                          <p className="text-stone-500 text-xs">LLM</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Data Storage - Position 180deg */}
                  <div
                    className="absolute top-1/2 left-1/2 w-64"
                    style={{
                      transform: 'translate(-50%, -50%) rotateY(180deg) translateZ(280px)',
                      transformStyle: 'preserve-3d',
                    }}
                  >
                    <div style={{ transform: `rotateY(${-archRotation - 180}deg)` }}>
                      <h3 className="text-sm font-semibold text-orange-400 mb-3 text-center">Data Storage</h3>
                      <div className="bg-zinc-800 rounded-xl p-4 border border-stone-700 space-y-3">
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <svg className="w-7 h-7 mx-auto mb-1" viewBox="0 0 80 80" fill="none">
                            <ellipse cx="40" cy="25" rx="30" ry="8" fill="#527FFF"/>
                            <path d="M10 25v30c0 4.4 13.4 8 30 8s30-3.6 30-8V25" fill="#2D72F0" opacity="0.7"/>
                            <ellipse cx="40" cy="55" rx="30" ry="8" fill="#1A5DC7"/>
                          </svg>
                          <p className="text-stone-200 text-xs font-semibold">DynamoDB</p>
                          <p className="text-stone-500 text-xs">NoSQL DB</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <svg className="w-7 h-7 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                            <path d="M4 4h16v16H4V4zm2 2v12h12V6H6z" fill="#8B5CF6"/>
                          </svg>
                          <p className="text-stone-200 text-xs font-semibold">FAISS</p>
                          <p className="text-stone-500 text-xs">4200+ vectors</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <svg className="w-7 h-7 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                            <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z" fill="#10A37F"/>
                          </svg>
                          <p className="text-stone-200 text-xs font-semibold">CLIP</p>
                          <p className="text-stone-500 text-xs">Similarity</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* DevOps - Position 240deg */}
                  <div
                    className="absolute top-1/2 left-1/2 w-64"
                    style={{
                      transform: 'translate(-50%, -50%) rotateY(240deg) translateZ(280px)',
                      transformStyle: 'preserve-3d',
                    }}
                  >
                    <div style={{ transform: `rotateY(${-archRotation - 240}deg)` }}>
                      <h3 className="text-sm font-semibold text-orange-400 mb-3 text-center">DevOps & CI/CD</h3>
                      <div className="bg-zinc-800 rounded-xl p-4 border border-stone-700 space-y-3">
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <Image src="https://www.vectorlogo.zone/logos/github/github-icon.svg" alt="GitHub" width={28} height={28} className="invert mx-auto mb-1"/>
                          <p className="text-stone-200 text-xs font-semibold">GitHub Actions</p>
                          <p className="text-stone-500 text-xs">CI/CD Pipeline</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <Image src="https://www.vectorlogo.zone/logos/amazon_aws/amazon_aws-icon.svg" alt="AWS Lambda" width={28} height={28} className="mx-auto mb-1"/>
                          <p className="text-stone-200 text-xs font-semibold">AWS SAM</p>
                          <p className="text-stone-500 text-xs">Deployment</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <Image src="https://www.vectorlogo.zone/logos/vercel/vercel-icon.svg" alt="Vercel" width={28} height={28} className="invert mx-auto mb-1"/>
                          <p className="text-stone-200 text-xs font-semibold">Vercel</p>
                          <p className="text-stone-500 text-xs">Auto Deploy</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Monitoring - Position 300deg */}
                  <div
                    className="absolute top-1/2 left-1/2 w-64"
                    style={{
                      transform: 'translate(-50%, -50%) rotateY(300deg) translateZ(280px)',
                      transformStyle: 'preserve-3d',
                    }}
                  >
                    <div style={{ transform: `rotateY(${-archRotation - 300}deg)` }}>
                      <h3 className="text-sm font-semibold text-orange-400 mb-3 text-center">Monitoring</h3>
                      <div className="bg-zinc-800 rounded-xl p-4 border border-stone-700 space-y-3">
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20">
                          <div className="flex items-center justify-center mb-2">
                            <Image src="https://www.vectorlogo.zone/logos/amazon_aws/amazon_aws-icon.svg" alt="CloudWatch" width={24} height={24} className="mr-2"/>
                            <p className="text-stone-200 text-xs font-semibold">CloudWatch</p>
                          </div>
                          <div className="grid grid-cols-3 gap-2 text-center mt-2">
                            <div>
                              <svg className="w-4 h-4 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                                <path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z" fill="#06B6D4"/>
                              </svg>
                              <p className="text-stone-400 text-[10px]">Logs</p>
                            </div>
                            <div>
                              <svg className="w-4 h-4 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                                <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z" fill="#10B981"/>
                              </svg>
                              <p className="text-stone-400 text-[10px]">Metrics</p>
                            </div>
                            <div>
                              <svg className="w-4 h-4 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                                <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z" fill="#EF4444"/>
                              </svg>
                              <p className="text-stone-400 text-[10px]">Alarms</p>
                            </div>
                          </div>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <Image src="https://avatars.githubusercontent.com/u/39938107?s=200&v=4" alt="MLflow" width={28} height={28} className="mx-auto mb-1 rounded"/>
                          <p className="text-stone-200 text-xs font-semibold">MLflow</p>
                          <p className="text-stone-500 text-xs">ML Tracking</p>
                        </div>
                        <div className="bg-zinc-900 p-3 rounded-lg border border-orange-400/20 text-center">
                          <svg className="w-7 h-7 mx-auto mb-1" viewBox="0 0 24 24" fill="none">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="#F59E0B"/>
                          </svg>
                          <p className="text-stone-200 text-xs font-semibold">Error Tracking</p>
                          <p className="text-stone-500 text-xs">Python Logs</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <style jsx>{`
                .perspective-\[1200px\] {
                  perspective: 1200px;
                }
              `}</style>
            </div>
            </div>
          </div>
        </div>


      </div>

      {/* Technology Stack Section */}
      <div id="about" className="py-24 px-8 bg-gradient-to-b from-stone-900 to-zinc-950">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-5xl font-bold text-center text-stone-100 mb-12 tracking-tight">Built With</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {/* PyTorch */}
            <div className="bg-zinc-900/50 backdrop-blur-md rounded-xl p-6 border border-stone-700/30 hover:border-orange-400/50 transition-all duration-300 flex flex-col items-center text-center group">
              <Image 
                src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" 
                alt="PyTorch"
                width={48}
                height={48}
                className="mb-3 group-hover:scale-110 transition-transform duration-300"
              />
              <h3 className="text-stone-200 font-semibold mb-1">PyTorch</h3>
              <p className="text-stone-500 text-xs">Deep Learning</p>
            </div>

            {/* AWS */}
            <div className="bg-zinc-900/50 backdrop-blur-md rounded-xl p-6 border border-stone-700/30 hover:border-orange-400/50 transition-all duration-300 flex flex-col items-center text-center group">
              <Image 
                src="https://www.vectorlogo.zone/logos/amazon_aws/amazon_aws-icon.svg" 
                alt="AWS"
                width={48}
                height={48}
                className="mb-3 group-hover:scale-110 transition-transform duration-300"
              />
              <h3 className="text-stone-200 font-semibold mb-1">AWS</h3>
              <p className="text-stone-500 text-xs">Cloud Infrastructure</p>
            </div>

            {/* Google Cloud */}
            <div className="bg-zinc-900/50 backdrop-blur-md rounded-xl p-6 border border-stone-700/30 hover:border-orange-400/50 transition-all duration-300 flex flex-col items-center text-center group">
              <Image 
                src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" 
                alt="Google Cloud"
                width={48}
                height={48}
                className="mb-3 group-hover:scale-110 transition-transform duration-300"
              />
              <h3 className="text-stone-200 font-semibold mb-1">Google Vision</h3>
              <p className="text-stone-500 text-xs">Vision API</p>
            </div>

            {/* Next.js */}
            <div className="bg-zinc-900/50 backdrop-blur-md rounded-xl p-6 border border-stone-700/30 hover:border-orange-400/50 transition-all duration-300 flex flex-col items-center text-center group">
              <Image 
                src="https://www.vectorlogo.zone/logos/nextjs/nextjs-icon.svg" 
                alt="Next.js"
                width={48}
                height={48}
                className="mb-3 group-hover:scale-110 transition-transform duration-300 invert"
              />
              <h3 className="text-stone-200 font-semibold mb-1">Next.js</h3>
              <p className="text-stone-500 text-xs">Frontend Framework</p>
            </div>

            {/* FastAPI */}
            <div className="bg-zinc-900/50 backdrop-blur-md rounded-xl p-6 border border-stone-700/30 hover:border-orange-400/50 transition-all duration-300 flex flex-col items-center text-center group">
              <svg className="w-12 h-12 mb-3 group-hover:scale-110 transition-transform duration-300" viewBox="0 0 24 24" fill="none">
                <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" fill="#009688" stroke="#009688" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <h3 className="text-stone-200 font-semibold mb-1">FastAPI</h3>
              <p className="text-stone-500 text-xs">Backend API</p>
            </div>

            {/* OpenAI */}
            <div className="bg-zinc-900/50 backdrop-blur-md rounded-xl p-6 border border-stone-700/30 hover:border-orange-400/50 transition-all duration-300 flex flex-col items-center text-center group">
              <svg className="w-12 h-12 mb-3 group-hover:scale-110 transition-transform duration-300" viewBox="0 0 24 24" fill="none">
                <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z" fill="#10A37F"/>
              </svg>
              <h3 className="text-stone-200 font-semibold mb-1">OpenAI CLIP</h3>
              <p className="text-stone-500 text-xs">Embeddings</p>
            </div>

            {/* DynamoDB */}
            <div className="bg-zinc-900/50 backdrop-blur-md rounded-xl p-6 border border-stone-700/30 hover:border-orange-400/50 transition-all duration-300 flex flex-col items-center text-center group">
              <svg className="w-12 h-12 mb-3 group-hover:scale-110 transition-transform duration-300" viewBox="0 0 80 80" fill="none">
                <path d="M40 10L10 25v30l30 15 30-15V25L40 10z" fill="#527FFF"/>
                <ellipse cx="40" cy="25" rx="30" ry="8" fill="#2D72F0"/>
                <path d="M10 25v30c0 4.4 13.4 8 30 8s30-3.6 30-8V25c0 4.4-13.4 8-30 8s-30-3.6-30-8z" fill="#2D72F0" opacity="0.7"/>
                <ellipse cx="40" cy="55" rx="30" ry="8" fill="#1A5DC7"/>
              </svg>
              <h3 className="text-stone-200 font-semibold mb-1">DynamoDB</h3>
              <p className="text-stone-500 text-xs">Database</p>
            </div>

            {/* Groq */}
            <div className="bg-zinc-900/50 backdrop-blur-md rounded-xl p-6 border border-stone-700/30 hover:border-orange-400/50 transition-all duration-300 flex flex-col items-center text-center group">
              <div className="w-12 h-12 mb-3 group-hover:scale-110 transition-transform duration-300 flex items-center justify-center bg-white rounded-lg">
                <span className="text-2xl font-bold text-black">G</span>
              </div>
              <h3 className="text-stone-200 font-semibold mb-1">Groq</h3>
              <p className="text-stone-500 text-xs">LLM Inference</p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-zinc-950 text-stone-500 py-12 px-8 border-t border-stone-800">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            {/* Brand */}
            <div>
              <Image 
                src="/images/logo_text.png" 
                alt="TripSaver" 
                width={150} 
                height={40} 
                className="h-8 w-auto mb-4"
              />
              <p className="text-sm text-stone-600">
                AI-powered travel planning made simple and free for everyone.
              </p>
            </div>

            {/* Quick Links */}
            <div>
              <h3 className="text-stone-300 font-semibold mb-4">Quick Links</h3>
              <ul className="space-y-2 text-sm">
                <li><a href="#features" className="hover:text-orange-400 transition-colors">Features</a></li>
                <li><a href="#how-it-works" className="hover:text-orange-400 transition-colors">Technology</a></li>
                <li><a href="#about" className="hover:text-orange-400 transition-colors">About</a></li>
                <li><Link href="/login" className="hover:text-orange-400 transition-colors">Get Started</Link></li>
              </ul>
            </div>

            {/* Connect */}
            <div>
              <h3 className="text-stone-300 font-semibold mb-4">Connect</h3>
              <ul className="space-y-2 text-sm">
                <li>
                  <a href="https://github.com/evanpetersen919" target="_blank" rel="noopener noreferrer" className="hover:text-orange-400 transition-colors">GitHub</a>
                </li>
                <li>
                  <a href="https://www.linkedin.com/in/evan-petersen-b93037386/" target="_blank" rel="noopener noreferrer" className="hover:text-orange-400 transition-colors">LinkedIn</a>
                </li>
                <li>
                  <a href="https://evanpetersen919.github.io/Portfolio/" target="_blank" rel="noopener noreferrer" className="hover:text-orange-400 transition-colors">Portfolio</a>
                </li>
              </ul>
            </div>
          </div>

          <div className="border-t border-stone-800 pt-8 text-center">
            <p className="text-sm font-light">© 2025 TripSaver. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
