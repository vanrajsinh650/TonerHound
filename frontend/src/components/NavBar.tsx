"use client";

import { FileSearch } from "lucide-react";

export function NavBar() {
  return (
    <nav className="border-b border-gray-200 bg-white px-6 py-3">
      <div className="mx-auto flex max-w-7xl items-center justify-between">
        <div className="flex items-center gap-2">
          <FileSearch className="h-6 w-6 text-gray-900" />
          <span className="text-lg font-semibold text-gray-900">TonerHound</span>
        </div>
        <a
          href="https://github.com/vanrajsinh650/TonerHound"
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
        >
          GitHub
        </a>
      </div>
    </nav>
  );
}
