import React from 'react';
// import { SunIcon, MoonIcon, TrashIcon } from '@heroicons/react/24/outline'; // Temporarily remove heroicons

const Header = ({ onClearCache }) => {
  return (
    <header className="bg-red-600 text-white px-4 py-6 shadow-md w-full h-28">
      <div className="container mx-auto flex justify-between items-center h-full">
        <h1 className="text-4xl md:text-5xl text-center flex-grow text-white">
          多源电力预测系统平台
        </h1>
        <div className="flex items-center space-x-3">
          <button
            onClick={onClearCache}
            title="清除服务器API缓存"
            className="p-2 rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-75 transition-colors text-sm font-medium"
          >
            {/* <TrashIcon className="h-6 w-6" /> */}
            清除缓存
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header; 