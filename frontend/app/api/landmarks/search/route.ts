import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get('q');

  if (!query || query.trim().length < 2) {
    return NextResponse.json({ landmarks: [] });
  }

  try {
    const landmarksPath = path.join(process.cwd(), '..', 'data', 'landmarks_unified.json');
    const fileContent = fs.readFileSync(landmarksPath, 'utf-8');
    const data = JSON.parse(fileContent);
    
    const queryLower = query.toLowerCase();
    const filtered = data.filter((landmark: any) => 
      landmark.name && landmark.name.toLowerCase().includes(queryLower)
    );

    const results = filtered.slice(0, 10).map((landmark: any) => ({
      name: landmark.name,
      latitude: landmark.latitude,
      longitude: landmark.longitude,
      country: landmark.country,
      description: landmark.description
    }));

    return NextResponse.json({ landmarks: results });
  } catch (error) {
    console.error('Error reading landmarks:', error);
    return NextResponse.json({ landmarks: [], error: 'Failed to load landmarks' }, { status: 500 });
  }
}
