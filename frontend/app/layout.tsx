import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Stock Recommender - Indian Market",
  description: "Research-grade AI-powered stock recommendation system for NSE/BSE using Multi-Agent AI and ML",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#0a0a0b] text-white antialiased">
        <div className="fixed inset-0 bg-gradient-to-br from-violet-900/20 via-transparent to-emerald-900/20 pointer-events-none" />
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  );
}
