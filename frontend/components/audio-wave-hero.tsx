"use client"

import { useEffect, useRef } from "react"

export default function AudioWaveHero() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions to match its display size
    const resizeCanvas = () => {
      const { width, height } = canvas.getBoundingClientRect()
      if (canvas.width !== width || canvas.height !== height) {
        const { devicePixelRatio: ratio = 1 } = window
        canvas.width = width * ratio
        canvas.height = height * ratio
        ctx.scale(ratio, ratio)
        return true
      }
      return false
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Animation variables
    let animationId: number
    const waveCount = 3
    const waves = Array.from({ length: waveCount }, (_, i) => ({
      frequency: 0.05 - i * 0.01,
      amplitude: 50 - i * 10,
      speed: 0.05 + i * 0.02,
      color: `rgba(99, 102, 241, ${0.7 - i * 0.2})`,
      offset: 0,
    }))

    // Animation function
    const animate = () => {
      const { width, height } = canvas.getBoundingClientRect()
      ctx.clearRect(0, 0, width, height)

      waves.forEach((wave) => {
        wave.offset += wave.speed

        ctx.beginPath()
        ctx.moveTo(0, height / 2)

        for (let x = 0; x < width; x++) {
          const y = Math.sin(x * wave.frequency + wave.offset) * wave.amplitude + height / 2
          ctx.lineTo(x, y)
        }

        ctx.strokeStyle = wave.color
        ctx.lineWidth = 3
        ctx.stroke()
      })

      animationId = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [])

  return (
    <section className="relative w-full h-[60vh] flex items-center justify-center bg-gradient-to-b from-slate-900 to-indigo-900 overflow-hidden">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      <div className="relative z-10 text-center px-4">
        <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">Audio & Video Platform</h1>
        <p className="text-xl text-white/80 max-w-2xl mx-auto">
          Non-verbal Sound Detection, Subtitle Generation, Supports 50+ Languages
        </p>
      </div>
    </section>
  )
}

