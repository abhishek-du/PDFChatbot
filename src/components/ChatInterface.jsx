/* ChatInterface.jsx */
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

const ChatInterface = () => {
  const [sessionId, setSessionId] = useState('');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  marked.setOptions({ breaks: true, gfm: true });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(scrollToBottom, [messages]);

  const formatResponse = (text) => DOMPurify.sanitize(marked.parse(text));

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post('http://localhost:5000/upload', formData);
      setSessionId(res.data.session_id);
      setMessages([{ text: `PDF "${file.name}" uploaded successfully.`, isUser: false, isSystem: true }]);
    } catch {
      alert('Upload error');
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !sessionId) return;
    setMessages(prev => [...prev, { text: inputMessage, isUser: true }]);
    setInputMessage('');
    setLoadingMessage(true);
    try {
      const res = await axios.post('http://localhost:5000/chat', { session_id: sessionId, question: inputMessage });
      setMessages(prev => [...prev, { text: res.data.answer, html: formatResponse(res.data.answer), isUser: false }]);
    } catch {
      setMessages(prev => [...prev, { text: 'Error getting response', isUser: false, isError: true }]);
    } finally {
      setLoadingMessage(false);
    }
  };

  const resetChat = () => {
    if (window.confirm('Clear chat and upload new PDF?')) {
      setSessionId(''); setMessages([]); setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="chat-container">
      {!sessionId ? (
        <div className="upload-panel">
          <h1>PDF Chatbot</h1>
          <form onSubmit={handleFileUpload}>
            <input type="file" accept=".pdf" ref={fileInputRef} onChange={handleFileChange} />
            <button type="submit" disabled={loading || !file}>{loading ? 'Processing...' : 'Upload PDF'}</button>
          </form>
        </div>
      ) : (
        <div className="chat-panel">
          <header>
            <h2>PDF Chat Assistant</h2>
            <button onClick={resetChat}>New PDF</button>
          </header>
          <main className="messages-area">
            {messages.length === 0 && <div className="empty-state">Start a conversation</div>}
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.isUser ? 'user' : msg.isSystem ? 'system' : msg.isError ? 'error' : 'bot'}`}>
                {msg.html ? <div dangerouslySetInnerHTML={{ __html: msg.html }} /> : <p>{msg.text}</p>}
              </div>
            ))}
            {loadingMessage && <div className="message bot loading">Thinking...</div>}
            <div ref={messagesEndRef} />
          </main>
          <footer>
            <input
              type="text"
              value={inputMessage}
              onChange={e => setInputMessage(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleSendMessage()}
              placeholder="Ask a question..."
              disabled={loadingMessage}
            />
            <button onClick={handleSendMessage} disabled={!inputMessage.trim() || loadingMessage}>Send</button>
          </footer>
        </div>
      )}
    </div>
  );
};

export default ChatInterface;

