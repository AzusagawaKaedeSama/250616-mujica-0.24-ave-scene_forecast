import React from 'react';
// import { SunIcon, MoonIcon, TrashIcon } from '@heroicons/react/24/outline'; // Temporarily remove heroicons

const Header = ({ onClearCache }) => {
  return (
    <header className="bg-neutral-900 text-white p-4 shadow-md flex justify-between items-center border-b border-neutral-700">
      <div className="flex items-center">
        <h1 className="text-xl font-bold text-red-500">Mujica-0.24-ave</h1>
      </div>
      <div className="flex items-center space-x-4">
        <button
          onClick={onClearCache}
          disabled  // Temporarily disable the button
          className="bg-neutral-700 text-neutral-300 px-4 py-2 rounded-md text-sm font-medium hover:bg-neutral-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-neutral-900 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {/* <TrashIcon className="h-6 w-6" /> */}
          清除缓存
        </button>
      </div>
    </header>
  );
};

export default Header; 