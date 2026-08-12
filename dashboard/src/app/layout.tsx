import type { Metadata } from "next";
import { Geist_Mono, Share_Tech_Mono } from "next/font/google";
import "./globals.css";

const shareTechMono = Share_Tech_Mono({
  variable: "--font-share-tech-mono",
  subsets: ["latin"],
  weight: "400",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "BlackICE-Mesh // Telemetry Terminal",
  description:
    "Polyglot adversarial ML defense mesh — Clean Accuracy, Robust Accuracy, ASR telemetry.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${shareTechMono.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="scanlines min-h-full">{children}</body>
    </html>
  );
}
