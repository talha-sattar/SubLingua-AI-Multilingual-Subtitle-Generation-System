"use client"

import type React from "react"
import { useState, useEffect } from "react" // Import useEffect if needed later, useState is key here
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, Video, AudioLines, X, Loader2, Download } from "lucide-react" // Added Loader2 and Download icons
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// --- Language, Output Type, Genre Options remain the same ---
const LANGUAGE_OPTIONS = [
  { value: "english", label: "English (Default)" },
  { value: "albanian", label: "Albanian" },
  { value: "amharic", label: "Amharic" },
  { value: "arabic", label: "Arabic" },
  { value: "armenian", label: "Armenian" },
  { value: "bengali", label: "Bengali" },
  { value: "bosnian", label: "Bosnian" },
  { value: "bulgarian", label: "Bulgarian" },
  { value: "burmese", label: "Burmese" },
  { value: "catalan", label: "Catalan" },
  { value: "chinese", label: "Chinese" },
  { value: "croatian", label: "Croatian" },
  { value: "czech", label: "Czech" },
  { value: "danish", label: "Danish" },
  { value: "dutch", label: "Dutch" },
  { value: "estonian", label: "Estonian" },
  { value: "finnish", label: "Finnish" },
  { value: "french", label: "French" },
  { value: "georgian", label: "Georgian" },
  { value: "german", label: "German" },
  { value: "greek", label: "Greek" },
  { value: "gujarati", label: "Gujarati" },
  { value: "hindi", label: "Hindi" },
  { value: "hungarian", label: "Hungarian" },
  { value: "icelandic", label: "Icelandic" },
  { value: "indonesian", label: "Indonesian" },
  { value: "italian", label: "Italian" },
  { value: "japanese", label: "Japanese" },
  { value: "kannada", label: "Kannada" },
  { value: "kazakh", label: "Kazakh" },
  { value: "korean", label: "Korean" },
  { value: "latvian", label: "Latvian" },
  { value: "lithuanian", label: "Lithuanian" },
  { value: "macedonian", label: "Macedonian" },
  { value: "malay", label: "Malay" },
  { value: "malayalam", label: "Malayalam" },
  { value: "marathi", label: "Marathi" },
  { value: "mongolian", label: "Mongolian" },
  { value: "norwegian", label: "Norwegian" },
  { value: "persian", label: "Persian" },
  { value: "polish", label: "Polish" },
  { value: "portuguese", label: "Portuguese" },
  { value: "punjabi", label: "Punjabi" },
  { value: "romanian", label: "Romanian" },
  { value: "russian", label: "Russian" },
  { value: "serbian", label: "Serbian" },
  { value: "slovak", label: "Slovak" },
  { value: "slovenian", label: "Slovenian" },
  { value: "somali", label: "Somali" },
  { value: "spanish", label: "Spanish" },
  { value: "swahili", label: "Swahili" },
  { value: "swedish", label: "Swedish" },
  { value: "tagalog", label: "Tagalog" },
  { value: "tamil", label: "Tamil" },
  { value: "telugu", label: "Telugu" },
  { value: "thai", label: "Thai" },
  { value: "turkish", label: "Turkish" },
  { value: "ukrainian", label: "Ukrainian" },
  { value: "urdu", label: "Urdu" },
  { value: "vietnamese", label: "Vietnamese" },
]

const OUTPUT_TYPE_OPTIONS = [
  { value: "srt", label: "SRT Subtitles (Default)" },
  { value: "vtt", label: "VTT Subtitles" },
]

const GENRE_OPTIONS = [
  { value: "g", label: "General (Default)" },
  { value: "1", label: "Action" },
  { value: "2", label: "Comedy" },
  { value: "3", label: "Dramatic" },
  { value: "4", label: "Horror" },
  { value: "5", label: "Romance" },
  { value: "6", label: "Sci-Fi" },
  { value: "7", label: "Thriller" },
  
  
  
  
  
]

const DELAY_OPTIONS = [
  { value: "none", label: "None (Default)" },
  { value: "1", label: "1 second" },
  { value: "2", label: "2 seconds" },
  { value: "3", label: "3 seconds" },
]

// Audio delay

// Handlers

// --- ---

// --- Backend URL Configuration ---
// It's often better to put this in environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000"
// --- ---

export default function MediaUploadSection() {
  const [activeTab, setActiveTab] = useState<"video" | "audio">("video")

  // --- Video States ---
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null)
  const [videoLanguage, setVideoLanguage] = useState<string>("english")
  const [videoOutputType, setVideoOutputType] = useState<string>("srt")
  const [videoGenre, setVideoGenre] = useState<string[]>(["g"])
  const [videoDelay, setVideoDelay] = useState<string>("none")
  const [isVideoLoading, setIsVideoLoading] = useState<boolean>(false) // New state for loading
  const [videoDownloadUrl, setVideoDownloadUrl] = useState<string | null>(null) // New state for download link
  const [videoError, setVideoError] = useState<string | null>(null) // New state for errors
  const [videoGenreDropdownOpen, setVideoGenreDropdownOpen] = useState(false)

  // --- Audio States ---
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null)
  const [audioLanguage, setAudioLanguage] = useState<string>("english")
  const [audioOutputType, setAudioOutputType] = useState<string>("srt")
  const [audioGenre, setAudioGenre] = useState<string[]>(["g"])
  const [audioDelay, setAudioDelay] = useState<string>("none")
  const [isAudioLoading, setIsAudioLoading] = useState<boolean>(false) // New state for loading
  const [audioDownloadUrl, setAudioDownloadUrl] = useState<string | null>(null) // New state for download link
  const [audioError, setAudioError] = useState<string | null>(null) // New state for errors
  const [audioGenreDropdownOpen, setAudioGenreDropdownOpen] = useState(false)

  // Revoke object URLs on component unmount or when URL changes
  useEffect(() => {
    const currentVideoUrl = videoPreviewUrl
    const currentAudioUrl = audioPreviewUrl
    return () => {
      if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl)
      if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl)
    }
  }, [videoPreviewUrl, audioPreviewUrl])

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith("video/")) {
      // Clear previous state before setting new file
      clearVideo()
      setVideoFile(file)
      const url = URL.createObjectURL(file)
      setVideoPreviewUrl(url)
      // Keep default selections or reset if preferred:
      // setVideoLanguage("english");
      // setVideoOutputType("vtt");
      // setVideoGenre("general");
    }
    // Reset input value to allow uploading the same file again
    e.target.value = ""
  }

  const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith("audio/")) {
      // Clear previous state before setting new file
      clearAudio()
      setAudioFile(file)
      const url = URL.createObjectURL(file)
      setAudioPreviewUrl(url)
      // Keep default selections or reset if preferred:
      // setAudioLanguage("english");
      // setAudioOutputType("vtt");
      // setAudioGenre("general");
    }
    // Reset input value to allow uploading the same file again
    e.target.value = ""
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]

    if (activeTab === "video" && file?.type.startsWith("video/")) {
      // Clear previous state before setting new file
      clearVideo()
      setVideoFile(file)
      const url = URL.createObjectURL(file)
      setVideoPreviewUrl(url)
      // Keep default selections or reset if preferred
    } else if (activeTab === "audio" && file?.type.startsWith("audio/")) {
      // Clear previous state before setting new file
      clearAudio()
      setAudioFile(file)
      const url = URL.createObjectURL(file)
      setAudioPreviewUrl(url)
      // Keep default selections or reset if preferred
    }
  }

  const clearVideo = () => {
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl)
    }
    setVideoFile(null)
    setVideoPreviewUrl(null)
    setIsVideoLoading(false) // Reset loading state
    setVideoDownloadUrl(null) // Reset download URL
    setVideoError(null) // Reset error state
    // Optionally reset dropdowns to default
    setVideoLanguage("english")
    setVideoOutputType("srt")
    setVideoGenre(["g"])
    setVideoDelay("none")
  }

  const clearAudio = () => {
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl)
    }
    setAudioFile(null)
    setAudioPreviewUrl(null)
    setIsAudioLoading(false) // Reset loading state
    setAudioDownloadUrl(null) // Reset download URL
    setAudioError(null) // Reset error state
    // Optionally reset dropdowns to default
    setAudioLanguage("english")
    setAudioOutputType("srt")
    setAudioGenre(["g"])
    setAudioDelay("none")
  }

  // --- Handler functions for dropdowns remain the same ---
  const handleVideoLanguageChange = (value: string) => setVideoLanguage(value)
  const handleAudioLanguageChange = (value: string) => setAudioLanguage(value)
  const handleVideoOutputTypeChange = (value: string) => setVideoOutputType(value)
  const handleAudioOutputTypeChange = (value: string) => setAudioOutputType(value)
  const handleVideoDelayChange = (value: string) => setVideoDelay(value)
  const handleAudioDelayChange = (value: string) => setAudioDelay(value)

  const handleVideoGenreChange = (value: string) => {
    if (videoGenre.includes(value)) {
      setVideoGenre(videoGenre.filter((genre) => genre !== value))
    } else {
      setVideoGenre([...videoGenre, value])
    }
  }
  const handleAudioGenreChange = (value: string) => {
    if (audioGenre.includes(value)) {
      setAudioGenre(audioGenre.filter((genre) => genre !== value))
    } else {
      setAudioGenre([...audioGenre, value])
    }
  }
  // --- ---

  const handleVideoSubmit = () => {
    if (!videoFile) {
      setVideoError("No video file selected.")
      console.error("No video file selected.")
      return
    }
    if (!videoLanguage || !videoOutputType || videoGenre.length === 0) {
      setVideoError("Please select language, genre, and output type.")
      console.error("Missing video options.")
      return
    }

    setIsVideoLoading(true) // Start loading
    setVideoDownloadUrl(null) // Clear previous download link
    setVideoError(null) // Clear previous errors

    const formData = new FormData()
    formData.append("file", videoFile)
    formData.append("activeTab", "video")
    formData.append("language", videoLanguage)
    formData.append("genre", videoGenre.join(","))
    formData.append("outputType", videoOutputType)
    formData.append("delay", videoDelay)

    fetch(`${API_BASE_URL}/api/upload`, {
      method: "POST",
      body: formData,
    })
      .then(async (response) => {
        if (response.ok) {
          console.log("Video uploaded successfully!")
          // Always download the single file in download folder:
          setVideoDownloadUrl(`${API_BASE_URL}/download/latest`)
        } else {
          const errorText = await response.text()
          console.error("Video upload failed:", response.statusText, errorText)
          setVideoError(`Upload failed: ${response.statusText}. ${errorText || ""}`)
        }
      })
      .catch((error) => {
        console.error("Error uploading video:", error)
        setVideoError(`An error occurred: ${error.message}`)
      })
      .finally(() => {
        setIsVideoLoading(false)
      })
    
  }

  const handleAudioSubmit = () => {
    if (!audioFile) {
      setAudioError("No audio file selected.")
      console.error("No audio file selected.")
      return
    }
    if (!audioLanguage || !audioOutputType || audioGenre.length === 0) {
      setAudioError("Please select language, genre, and output type.")
      console.error("Missing audio options.")
      return
    }

    setIsAudioLoading(true) // Start loading
    setAudioDownloadUrl(null) // Clear previous download link
    setAudioError(null) // Clear previous errors

    const formData = new FormData()
    formData.append("file", audioFile)
    formData.append("activeTab", "audio")
    formData.append("language", audioLanguage)
    formData.append("genre", audioGenre.join(","))
    formData.append("outputType", audioOutputType)
    formData.append("delay", audioDelay)

    fetch(`${API_BASE_URL}/api/upload`, {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (response.ok) {
          console.log("Audio uploaded successfully!")
          // Always download the single file in download folder:
          setAudioDownloadUrl(`${API_BASE_URL}/download/latest`)
        } else {
          return response.text().then((errorText) => {
            console.error("Audio upload failed:", response.statusText, errorText)
            setAudioError(`Upload failed: ${response.statusText}. ${errorText || ""}`)
          })
        }
      })
      .catch((error) => {
        console.error("Error uploading audio:", error)
        setAudioError(`An error occurred: ${error.message}`)
      })
      .finally(() => {
        setIsAudioLoading(false)
      })
    
  }

  // Handle clicks outside the dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      // Video genre dropdown
      if (videoGenreDropdownOpen) {
        const element = document.getElementById("video-genre-dropdown")
        if (element && !element.contains(event.target as Node)) {
          setVideoGenreDropdownOpen(false)
        }
      }

      // Audio genre dropdown
      if (audioGenreDropdownOpen) {
        const element = document.getElementById("audio-genre-dropdown")
        if (element && !element.contains(event.target as Node)) {
          setAudioGenreDropdownOpen(false)
        }
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [videoGenreDropdownOpen, audioGenreDropdownOpen])

  return (
    <section className="w-full py-16 px-4 bg-slate-50">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold mb-4">Upload Your Media</h2>
          <p className="text-slate-600 max-w-2xl mx-auto">
            Drag and drop or browse to upload your video or audio files. We support most common formats.
          </p>
        </div>

        <Tabs
          defaultValue="video"
          className="w-full"
          onValueChange={(value) => setActiveTab(value as "video" | "audio")}
        >
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger
              value="video"
              className="flex items-center gap-2 cursor-pointer"
              disabled={isVideoLoading || isAudioLoading}
            >
              <Video className="h-4 w-4" />
              <span>Video Upload</span>
            </TabsTrigger>
            <TabsTrigger
              value="audio"
              className="flex items-center gap-2 cursor-pointer"
              disabled={isVideoLoading || isAudioLoading}
            >
              <AudioLines className="h-4 w-4" />
              <span>Audio Upload</span>
            </TabsTrigger>
          </TabsList>

          {/* Video Tab Content */}
          <TabsContent value="video">
            <Card>
              <CardHeader>
                <CardTitle>Upload Video</CardTitle>
                <CardDescription>Upload your video files in MP4, WebM, or MOV format.</CardDescription>
              </CardHeader>

              {/* Display File Info Tab if file exists and not showing download options */}
              {videoFile && !videoDownloadUrl && (
                <div className="mx-6 mt-2 mb-4">
                  <div className="flex items-center">
                    <div className="flex items-center gap-2 bg-slate-100 px-4 py-2 rounded-t-lg border border-slate-200">
                      <Video className="h-4 w-4 text-slate-500" />
                      <div className="flex flex-col">
                        <span className="font-medium text-sm truncate max-w-[200px]">{videoFile.name}</span>
                        <span className="text-xs text-slate-500">{(videoFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                      </div>
                      <button
                        onClick={clearVideo}
                        className="ml-2 text-slate-400 hover:text-slate-600 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={isVideoLoading} // Disable clear while loading
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              )}

              <CardContent>
                {/* Initial Upload Area */}
                {!videoPreviewUrl && !videoDownloadUrl && (
                  <div
                    className={`border-2 border-dashed border-slate-300 rounded-lg p-12 text-center transition-colors ${isVideoLoading || isAudioLoading ? "bg-slate-100 cursor-not-allowed opacity-70" : "hover:bg-slate-100 cursor-pointer"}`}
                    onDragOver={!isVideoLoading && !isAudioLoading ? handleDragOver : undefined}
                    onDrop={!isVideoLoading && !isAudioLoading ? handleDrop : undefined}
                    onClick={() =>
                      !isVideoLoading && !isAudioLoading && document.getElementById("video-upload")?.click()
                    }
                  >
                    <Upload className="h-12 w-12 mx-auto mb-4 text-slate-400" />
                    <p className="text-slate-600 mb-2">Drag and drop your video here or click to browse</p>
                    <p className="text-slate-400 text-sm">Supports MP4, MOV, MOV up to 2 GB</p>
                    <input
                      type="file"
                      id="video-upload"
                      className="hidden"
                      accept="video/*"
                      onChange={handleVideoUpload}
                      disabled={isVideoLoading || isAudioLoading} // Disable input during any loading
                    />
                  </div>
                )}

                {/* Options Area (shown when preview exists, not loading, and no download link yet) */}
                {videoPreviewUrl && !isVideoLoading && !videoDownloadUrl && (
                  <div className="space-y-6">
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        {/* Language Dropdown */}
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-slate-700">Select Language</label>
                          <Select value={videoLanguage} onValueChange={handleVideoLanguageChange}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select language" />
                            </SelectTrigger>
                            <SelectContent>
                              {LANGUAGE_OPTIONS.map((option) => (
                                <SelectItem key={option.value} value={option.value}>
                                  {option.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        {/* Genre Multi-Select Dropdown */}
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-slate-700">Select Genre(s)</label>
                          <div className="relative">
                            <button
                              type="button"
                              id="video-genre-button"
                              onClick={() => setVideoGenreDropdownOpen(!videoGenreDropdownOpen)}
                              className="flex w-full justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                              <span className="truncate">
                                {videoGenre.length === 0
                                  ? "Select genre(s)"
                                  : videoGenre.length === 1
                                    ? GENRE_OPTIONS.find((g) => g.value === videoGenre[0])?.label
                                    : `${videoGenre.length} genres selected`}
                              </span>
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                width="24"
                                height="24"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="ml-2 h-4 w-4 shrink-0 opacity-50"
                              >
                                <path d="m6 9 6 6 6-6" />
                              </svg>
                            </button>
                            {videoGenreDropdownOpen && (
                              <div
                                id="video-genre-dropdown"
                                className="absolute z-10 mt-1 w-full rounded-md border border-input bg-background shadow-md"
                              >
                                <div className="max-h-[200px] overflow-auto p-1">
                                  {GENRE_OPTIONS.map((option) => (
                                    <div
                                      key={option.value}
                                      className="flex items-center space-x-2 rounded px-2 py-1.5 hover:bg-accent"
                                    >
                                      <input
                                        type="checkbox"
                                        id={`video-genre-${option.value}`}
                                        checked={videoGenre.includes(option.value)}
                                        onChange={(e) => {
                                          if (e.target.checked) {
                                            setVideoGenre([...videoGenre, option.value])
                                          } else {
                                            setVideoGenre(videoGenre.filter((genre) => genre !== option.value))
                                          }
                                        }}
                                        className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                                      />
                                      <label
                                        htmlFor={`video-genre-${option.value}`}
                                        className="w-full cursor-pointer text-sm"
                                      >
                                        {option.label}
                                      </label>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="space-y-2">
                          <label className="text-sm font-medium text-slate-700">Select Delay</label>
                          <Select value={videoDelay} onValueChange={handleVideoDelayChange}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select delay" />
                            </SelectTrigger>
                            <SelectContent>
                              {DELAY_OPTIONS.map((opt) => (
                                <SelectItem key={opt.value} value={opt.value}>
                                  {opt.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        {/* Output Type Dropdown */}
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-slate-700">Select Output Type</label>
                          <Select value={videoOutputType} onValueChange={handleVideoOutputTypeChange}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select output type" />
                            </SelectTrigger>
                            <SelectContent>
                              {OUTPUT_TYPE_OPTIONS.map((option) => (
                                <SelectItem key={option.value} value={option.value}>
                                  {option.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      {/* Error Message Display */}
                      {videoError && <p className="text-sm text-red-600 mt-2">{videoError}</p>}

                      {/* Submit Button */}
                      <div className="flex justify-end mt-6 ">
                        <Button
                          size="sm"
                          className="cursor-pointer"
                          disabled={!videoLanguage || !videoOutputType || videoGenre.length === 0}
                          onClick={handleVideoSubmit}
                        >
                          Submit
                        </Button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Loading State Display */}
                {isVideoLoading && (
                  <div className="flex justify-center items-center flex-col space-y-4 p-8">
                    <Loader2 className="h-10 w-10 text-slate-500 animate-spin" />
                    <p className="text-slate-600 text-lg font-medium">Processing video...</p>
                    <p className="text-slate-500 text-sm">This might take a few moments.</p>
                  </div>
                )}

                {/* Download/Reset Area (shown after successful processing) */}
                {videoDownloadUrl && !isVideoLoading && (
                  <div className="space-y-4 text-center p-6 bg-green-50 border border-green-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-green-800">Processing Complete!</h3>
                    {/* Error Message Display if download link failed but processing technically succeeded */}
                    {videoError && <p className="text-sm text-red-600 mt-2">{videoError}</p>}
                    <div className="flex justify-center items-center space-x-4 mt-4">
                      {!videoError &&
                        videoDownloadUrl && ( // Only show download if no error and URL exists
                          <Button size="sm" asChild>
                            <a href={videoDownloadUrl} download>
                              {" "}
                              {/* Use download attribute */}
                              <Download className="mr-2 h-4 w-4" /> Download File
                            </a>
                          </Button>
                        )}
                      <Button size="sm" variant="outline" onClick={clearVideo}>
                        <Upload className="mr-2 h-4 w-4" /> Upload Another Video
                      </Button>
                    </div>
                  </div>
                )}

                {/* Error Display Area (shown if upload/processing failed before download link generated) */}
                {!isVideoLoading && !videoDownloadUrl && videoError && videoPreviewUrl && (
                  <div className="space-y-4 text-center p-6 bg-red-50 border border-red-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-red-800">Upload Failed</h3>
                    <p className="text-sm text-red-600">{videoError}</p>
                    <div className="flex justify-center items-center space-x-4 mt-4">
                      <Button size="sm" variant="outline" onClick={clearVideo}>
                        Try Again
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Audio Tab Content (Similar structure to Video) */}
          <TabsContent value="audio">
            <Card>
              <CardHeader>
                <CardTitle>Upload Audio</CardTitle>
                <CardDescription>Upload your audio files in MP3, WAV, or OGG format.</CardDescription>
              </CardHeader>

              {/* Display File Info Tab if file exists and not showing download options */}
              {audioFile && !audioDownloadUrl && (
                <div className="mx-6 mt-2 mb-4">
                  <div className="flex items-center">
                    <div className="flex items-center gap-2 bg-slate-100 px-4 py-2 rounded-t-lg border border-slate-200">
                      <AudioLines className="h-4 w-4 text-slate-500" />
                      <div className="flex flex-col">
                        <span className="font-medium text-sm truncate max-w-[200px]">{audioFile.name}</span>
                        <span className="text-xs text-slate-500">{(audioFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                      </div>
                      <button
                        onClick={clearAudio}
                        className="ml-2 text-slate-400 hover:text-slate-600 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={isAudioLoading} // Disable clear while loading
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              )}

              <CardContent>
                {/* Initial Upload Area */}
                {!audioPreviewUrl && !audioDownloadUrl && (
                  <div
                    className={`border-2 border-dashed border-slate-300 rounded-lg p-12 text-center transition-colors ${isVideoLoading || isAudioLoading ? "bg-slate-100 cursor-not-allowed opacity-70" : "hover:bg-slate-100 cursor-pointer"}`}
                    onDragOver={!isVideoLoading && !isAudioLoading ? handleDragOver : undefined}
                    onDrop={!isVideoLoading && !isAudioLoading ? handleDrop : undefined}
                    onClick={() =>
                      !isVideoLoading && !isAudioLoading && document.getElementById("audio-upload")?.click()
                    }
                  >
                    <Upload className="h-12 w-12 mx-auto mb-4 text-slate-400" />
                    <p className="text-slate-600 mb-2">Drag and drop your audio here or click to browse</p>
                    <p className="text-slate-400 text-sm">Supports MP3, WAV, M4 up to 2 GB</p>
                    <input
                      type="file"
                      id="audio-upload"
                      className="hidden"
                      accept="audio/*"
                      onChange={handleAudioUpload}
                      disabled={isVideoLoading || isAudioLoading} // Disable input during any loading
                    />
                  </div>
                )}

                {/* Options Area (shown when preview exists, not loading, and no download link yet) */}
                {audioPreviewUrl && !isAudioLoading && !audioDownloadUrl && (
                  <div className="space-y-6">
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        {/* Language Dropdown */}
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-slate-700">Select Language</label>
                          <Select value={audioLanguage} onValueChange={handleAudioLanguageChange}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select language" />
                            </SelectTrigger>
                            <SelectContent>
                              {LANGUAGE_OPTIONS.map((option) => (
                                <SelectItem key={option.value} value={option.value}>
                                  {option.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        {/* Genre Multi-Select Dropdown */}
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-slate-700">Select Genre(s)</label>
                          <div className="relative">
                            <button
                              type="button"
                              id="audio-genre-button"
                              onClick={() => setAudioGenreDropdownOpen(!audioGenreDropdownOpen)}
                              className="flex w-full justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                              <span className="truncate">
                                {audioGenre.length === 0
                                  ? "Select genre(s)"
                                  : audioGenre.length === 1
                                    ? GENRE_OPTIONS.find((g) => g.value === audioGenre[0])?.label
                                    : `${audioGenre.length} genres selected`}
                              </span>
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                width="24"
                                height="24"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="ml-2 h-4 w-4 shrink-0 opacity-50"
                              >
                                <path d="m6 9 6 6 6-6" />
                              </svg>
                            </button>
                            {audioGenreDropdownOpen && (
                              <div
                                id="audio-genre-dropdown"
                                className="absolute z-10 mt-1 w-full rounded-md border border-input bg-background shadow-md"
                              >
                                <div className="max-h-[200px] overflow-auto p-1">
                                  {GENRE_OPTIONS.map((option) => (
                                    <div
                                      key={option.value}
                                      className="flex items-center space-x-2 rounded px-2 py-1.5 hover:bg-accent"
                                    >
                                      <input
                                        type="checkbox"
                                        id={`audio-genre-${option.value}`}
                                        checked={audioGenre.includes(option.value)}
                                        onChange={(e) => {
                                          if (e.target.checked) {
                                            setAudioGenre([...audioGenre, option.value])
                                          } else {
                                            setAudioGenre(audioGenre.filter((genre) => genre !== option.value))
                                          }
                                        }}
                                        className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                                      />
                                      <label
                                        htmlFor={`audio-genre-${option.value}`}
                                        className="w-full cursor-pointer text-sm"
                                      >
                                        {option.label}
                                      </label>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="space-y-2">
                          <label className="text-sm font-medium text-slate-700">Select Delay</label>
                          <Select value={audioDelay} onValueChange={handleAudioDelayChange}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select delay" />
                            </SelectTrigger>
                            <SelectContent>
                              {DELAY_OPTIONS.map((opt) => (
                                <SelectItem key={opt.value} value={opt.value}>
                                  {opt.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        {/* Output Type Dropdown */}
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-slate-700">Select Output Type</label>
                          <Select value={audioOutputType} onValueChange={handleAudioOutputTypeChange}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select output type" />
                            </SelectTrigger>
                            <SelectContent>
                              {OUTPUT_TYPE_OPTIONS.map((option) => (
                                <SelectItem key={option.value} value={option.value}>
                                  {option.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      {/* Error Message Display */}
                      {audioError && <p className="text-sm text-red-600 mt-2">{audioError}</p>}

                      {/* Submit Button */}
                      <div className="flex justify-end mt-6 ">
                        <Button
                          size="sm"
                          className="cursor-pointer"
                          disabled={!audioLanguage || !audioOutputType || audioGenre.length === 0}
                          onClick={handleAudioSubmit}
                        >
                          Submit
                        </Button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Loading State Display */}
                {isAudioLoading && (
                  <div className="flex justify-center items-center flex-col space-y-4 p-8">
                    <Loader2 className="h-10 w-10 text-slate-500 animate-spin" />
                    <p className="text-slate-600 text-lg font-medium">Processing audio...</p>
                    <p className="text-slate-500 text-sm">This might take a few moments.</p>
                  </div>
                )}

                {/* Download/Reset Area (shown after successful processing) */}
                {audioDownloadUrl && !isAudioLoading && (
                  <div className="space-y-4 text-center p-6 bg-green-50 border border-green-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-green-800">Processing Complete!</h3>
                    {/* Error Message Display if download link failed but processing technically succeeded */}
                    {audioError && <p className="text-sm text-red-600 mt-2">{audioError}</p>}
                    <div className="flex justify-center items-center space-x-4 mt-4">
                      {!audioError &&
                        audioDownloadUrl && ( // Only show download if no error and URL exists
                          <Button size="sm" asChild>
                            {/* Assumes backend provides a relative or absolute URL */}
                            <a href={audioDownloadUrl} download>
                              {" "}
                              {/* Use download attribute */}
                              <Download className="mr-2 h-4 w-4" /> Download File
                            </a>
                          </Button>
                        )}
                      <Button size="sm" variant="outline" onClick={clearAudio}>
                        <Upload className="mr-2 h-4 w-4" /> Upload Another Audio
                      </Button>
                    </div>
                  </div>
                )}

                {/* Error Display Area (shown if upload/processing failed before download link generated) */}
                {!isAudioLoading && !audioDownloadUrl && audioError && audioPreviewUrl && (
                  <div className="space-y-4 text-center p-6 bg-red-50 border border-red-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-red-800">Upload Failed</h3>
                    <p className="text-sm text-red-600">{audioError}</p>
                    <div className="flex justify-center items-center space-x-4 mt-4">
                      <Button size="sm" variant="outline" onClick={clearAudio}>
                        Try Again
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  )
}
