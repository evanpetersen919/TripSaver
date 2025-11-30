'use client';

import Link from 'next/link';
import Image from 'next/image';

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section with Background Image */}
      <div className="relative py-32 px-8 min-h-screen bg-gradient-to-br from-orange-950 via-zinc-900 to-stone-900">
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
            className="group relative inline-block overflow-hidden bg-gradient-to-r from-orange-500 via-red-600 to-amber-600 text-white px-12 py-5 rounded-2xl text-lg font-semibold hover:scale-110 hover:shadow-[0_0_40px_rgba(249,115,22,0.6)] transition-all duration-300 shadow-2xl mb-16 animate-pulse hover:animate-none"
          >
            <span className="relative z-10 flex items-center gap-2">
              Start Planning Now
              <svg className="w-5 h-5 group-hover:translate-x-2 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </span>
            <div className="absolute inset-0 bg-gradient-to-r from-amber-600 via-red-600 to-orange-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
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
          <div 
            className="relative z-10 bg-zinc-900 bg-opacity-40 backdrop-blur-md rounded-3xl p-12 border border-stone-700 border-opacity-30 mt-48"
            style={{ 
              backdropFilter: 'blur(20px) saturate(180%)', 
              WebkitBackdropFilter: 'blur(20px) saturate(180%)',
              backgroundColor: 'rgba(24, 24, 27, 0.4)'
            } as React.CSSProperties}
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
              {/* Feature 1: Upload & Identify */}
              <div className="flex flex-col items-center text-center group cursor-pointer hover:-translate-y-3 transition-all duration-300">
                <div className="w-24 h-24 mb-4 rounded-2xl overflow-hidden border border-orange-400 border-opacity-40 group-hover:border-opacity-100 group-hover:scale-110 group-hover:shadow-[0_0_30px_rgba(251,146,60,0.5)] transition-all duration-300">
                  <Image
                    src="/images/upload_icon.png"
                    alt="Upload & Identify"
                    width={96}
                    height={96}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                </div>
                <h3 className="text-xl font-semibold text-stone-100 mb-2 group-hover:text-orange-400 transition-colors duration-300">Upload & Identify</h3>
                <p className="text-stone-300 text-sm leading-relaxed group-hover:text-stone-100 transition-colors duration-300">
                  Instantly recognize landmarks from any travel photo or screenshot.
                </p>
              </div>

              {/* Feature 2: Visual Similarity Search */}
              <div className="flex flex-col items-center text-center group cursor-pointer hover:-translate-y-3 transition-all duration-300">
                <div className="w-24 h-24 mb-4 rounded-2xl overflow-hidden border border-orange-400 border-opacity-40 group-hover:border-opacity-100 group-hover:scale-110 group-hover:shadow-[0_0_30px_rgba(251,146,60,0.5)] transition-all duration-300">
                  <Image
                    src="/images/search_icon.png"
                    alt="Visual Similarity Search"
                    width={96}
                    height={96}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                </div>
                <h3 className="text-xl font-semibold text-stone-100 mb-2 group-hover:text-orange-400 transition-colors duration-300">Visual Similarity Search</h3>
                <p className="text-stone-300 text-sm leading-relaxed group-hover:text-stone-100 transition-colors duration-300">
                  Find visually similar landmarks across a database of 200K+ locations.
                </p>
              </div>

              {/* Feature 3: Smart Recommendations */}
              <div className="flex flex-col items-center text-center group cursor-pointer hover:-translate-y-3 transition-all duration-300">
                <div className="w-24 h-24 mb-4 rounded-2xl overflow-hidden border border-orange-400 border-opacity-40 group-hover:border-opacity-100 group-hover:scale-110 group-hover:shadow-[0_0_30px_rgba(251,146,60,0.5)] transition-all duration-300">
                  <Image
                    src="/images/location_icon.png"
                    alt="Smart Recommendations"
                    width={96}
                    height={96}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                </div>
                <h3 className="text-xl font-semibold text-stone-100 mb-2 group-hover:text-orange-400 transition-colors duration-300">Smart Recommendations</h3>
                <p className="text-stone-300 text-sm leading-relaxed group-hover:text-stone-100 transition-colors duration-300">
                  Receive nearby landmark suggestions tailored to your detected locations.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Blocks - Vertically Stacked */}
      <div className="py-24 px-8 bg-gradient-to-b from-stone-900 via-zinc-900 to-stone-900">
        <div className="max-w-4xl mx-auto space-y-56">
          {/* Training */}
          <div className="border-l-2 border-orange-400 border-opacity-40 pl-12 hover:border-opacity-100 hover:pl-14 hover:-mr-2 transition-all duration-300 group">
            <div className="min-w-0">
            <h2 className="text-5xl font-semibold text-stone-100 mb-6 tracking-tight group-hover:text-orange-400 transition-colors duration-300">Training</h2>
            <p className="text-stone-200 text-lg leading-relaxed mb-8 font-normal">
              ResNet-50 was trained on RTX 4080 using Google Landmarks Dataset v2, starting with ImageNet pretrained weights and fine tuning on 500 landmark classes filtered from the full 200K+ class dataset. Training employed standard augmentations including RandAugment, MixUp, and CutMix, plus custom social media augmentation that synthesizes Instagram/TikTok UI overlays to improve robustness on screenshot inputs.
            </p>
            <ul className="space-y-5">
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Mixed precision FP16 training on RTX 4080 with AdamW optimizer and cosine annealing schedule. Gradient accumulation over 4 steps and early stopping prevent overfitting on imbalanced landmark classes</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Social media augmentation applies Instagram/TikTok overlays (profile bars, story circles, filters) at 30% probability during training, enabling the model to maintain accuracy when users upload screenshots rather than clean photographs</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Google Landmarks v2 provides 5M+ images across 200K+ landmarks worldwide. The dataset was filtered to 500 classes with sufficient training samples and geographic diversity for practical travel applications</span>
              </li>
            </ul>
            </div>
          </div>

          {/* How It Works */}
          <div className="border-l-2 border-orange-400 border-opacity-40 pl-12 hover:border-opacity-100 hover:pl-14 hover:-mr-2 transition-all duration-300 group">
            <div className="min-w-0">
            <h2 className="text-5xl font-semibold text-stone-100 mb-6 tracking-tight group-hover:text-orange-400 transition-colors duration-300">How It Works</h2>
            <p className="text-stone-200 text-lg leading-relaxed mb-8 font-normal">
              The system executes three models in parallel: ResNet-50 for landmark classification, CLIP for visual similarity search, and LLaVA for natural language descriptions. Uploaded images are preprocessed to remove social media UI elements, then distributed to all three models simultaneously. Results are combined using weighted confidence scores to achieve accurate landmark identification.
            </p>
            <ul className="space-y-5">
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Preprocessing pipeline applies template matching and edge detection to remove Instagram/TikTok UI elements. Images are normalized to 224x224 resolution using ImageNet mean and standard deviation</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Three models execute in parallel via asyncio: ResNet-50 performs classification, CLIP searches a FAISS index of 200K embeddings for visual similarity, and LLaVA generates descriptive text. Total inference latency under 3 seconds on GPU</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Model predictions are weighted by confidence scores (ResNet softmax, CLIP distances, LLaVA probabilities) and geocoded to GPS coordinates. The system recommends nearby landmarks using spatial indexing with KD-Tree for fast proximity search</span>
              </li>
            </ul>
            </div>
          </div>

          {/* Deployment */}
          <div className="border-l-2 border-orange-400 border-opacity-40 pl-12 hover:border-opacity-100 hover:pl-14 hover:-mr-2 transition-all duration-300 group">
            <div className="min-w-0">
            <h2 className="text-5xl font-semibold text-stone-100 mb-6 tracking-tight group-hover:text-orange-400 transition-colors duration-300">Deployment</h2>
            <p className="text-stone-200 text-lg font-normal leading-relaxed mb-8">
              The backend runs on AWS Lambda with FastAPI, using DynamoDB for landmark metadata and itinerary storage. Due to Lambda's 50MB deployment package limit, model inference is hosted separately on Hugging Face Spaces with GPU acceleration. CloudWatch provides monitoring for API latency and error tracking across the infrastructure.
            </p>
            <ul className="space-y-5">
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">AWS Lambda hosts the FastAPI backend behind API Gateway. DynamoDB stores landmark metadata including names, GPS coordinates, and descriptions for fast retrieval</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">Model inference executes on Hugging Face Spaces GPU instances. Lambda functions call the HF endpoint via REST API and process returned predictions for client consumption</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-4 mt-1 text-xl">•</span>
                <span className="text-stone-200 text-lg font-normal leading-relaxed">CloudWatch monitors API performance with custom metrics tracking request latency and error rates. The Next.js frontend is deployed separately and communicates with Lambda via API Gateway</span>
              </li>
            </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-stone-900 text-stone-500 py-10 px-8 border-t border-stone-800">
        <div className="max-w-6xl mx-auto text-center">
          <p className="mb-2 font-light">© 2025 TripSaver. All rights reserved.</p>
          <p className="text-sm text-stone-600 font-light">
            AI-powered travel planning made simple and free for everyone.
          </p>
        </div>
      </footer>
    </div>
  );
}
