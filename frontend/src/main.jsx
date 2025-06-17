import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './app';
import './styles/main.css'; // 恢复导入 main.css

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);