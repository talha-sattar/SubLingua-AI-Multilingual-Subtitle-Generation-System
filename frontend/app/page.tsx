import AudioWaveHero from "@/components/audio-wave-hero"
import MediaUploadSection from "@/components/media-upload-section"

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col">
      <AudioWaveHero />
      <MediaUploadSection />
    </main>
  )
}

