import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SERS-Insight Platform | AI-Powered Raman Spectroscopy Analysis",
  description: "A modular, extensible web-based platform for Surface-Enhanced Raman Spectroscopy (SERS) analysis with intelligent framework selection, automated modelling, and community-driven workflows.",
  keywords: ["SERS", "Raman Spectroscopy", "Biomedical Analysis", "AI", "Machine Learning", "Spectral Analysis"],
  authors: [{ name: "SERS-Insight Team" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased min-h-screen bg-white`}>
        {children}
      </body>
    </html>
  );
}
