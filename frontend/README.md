# TripSaver Frontend
### Next.js 16 Web Application for AI-Powered Landmark Recognition

A modern, responsive web interface for the TripSaver AI platform. Built with Next.js 16 (React 19), Tailwind CSS, and deployed on Vercel.

## Features

- **Image Upload & Recognition**: Drag-and-drop or click to upload travel photos for instant landmark identification
- **Visual Similarity Search**: Find visually similar landmarks across 4,248 locations
- **Real-Time Results**: Fast API integration with AWS Lambda backend
- **Interactive Architecture Display**: 3D carousel showing system components (drag to explore)
- **User Authentication**: Secure JWT-based login and signup
- **Itinerary Management**: Save and organize discovered landmarks
- **Responsive Design**: Optimized for mobile, tablet, and desktop
- **Dark Theme**: Modern UI with gradient accents and smooth animations

## Tech Stack

- **Framework**: Next.js 16 (React 19)
- **Styling**: Tailwind CSS
- **Deployment**: Vercel (auto-deploy on GitHub push)
- **API Integration**: RESTful calls to AWS API Gateway
- **State Management**: React hooks (useState, useEffect)
- **Image Optimization**: Next.js Image component
- **Authentication**: JWT tokens stored in localStorage

## Local Setup

### Prerequisites
- Node.js 18+
- npm or yarn

### Run Locally

1. **Install dependencies**
```bash
npm install
```

2. **Set up environment**
Create `.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
# Or use deployed backend:
# NEXT_PUBLIC_API_URL=https://your-api-gateway-url.amazonaws.com/prod
```

3. **Run dev server**
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000)

### Production Build
```bash
npm run build
npm start
```

## Deployment

**Vercel auto-deploy:**
1. Push to `main` branch
2. Vercel builds and deploys automatically
3. Set `NEXT_PUBLIC_API_URL` in Vercel dashboard

## Project Structure

```
frontend/
├── app/
│   ├── page.tsx           # Homepage with features and technical deep dive
│   ├── login/page.tsx     # Login page
│   ├── signup/page.tsx    # Signup page
│   ├── overview/page.tsx  # Main dashboard after login
│   └── layout.tsx         # Root layout
├── components/
│   ├── UploadSection.tsx  # Image upload component
│   ├── ResultsDisplay.tsx # Prediction results display
│   └── ItineraryList.tsx  # Saved landmarks list
├── lib/
│   ├── api.ts             # API client functions
│   └── destination-images.ts  # Landmark image mappings
├── public/
│   ├── images/            # Static images
│   └── videos/            # Demo videos
└── tailwind.config.ts     # Tailwind configuration
```

## Key Pages

### Homepage (`/`)
- Hero section with project overview
- Feature showcase with image examples
- Technical deep dive (training, detection flow, architecture)
- 3D draggable architecture carousel
- CTA to get started

### Overview (`/overview`)
- Main application interface
- Image upload and landmark detection
- Visual similarity search results
- Itinerary management
- User profile

### Authentication
- `/login` - JWT-based authentication
- `/signup` - New user registration
- Protected routes redirect to login if not authenticated



## Development Notes

- Uses App Router (Next.js 14+)
- Client-side rendering for interactive components (`'use client'`)
- Image optimization with Next.js Image component
- Tailwind CSS for utility-first styling
- Custom animations and transitions
- Responsive breakpoints for mobile/tablet/desktop

### Useful Commands
```bash
npm run dev          # Start dev server
npm run build        # Production build
npm run start        # Run production build
npm run lint         # Run ESLint
```

## License

All Rights Reserved © 2025 Evan Petersen
