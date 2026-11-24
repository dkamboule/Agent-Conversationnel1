import { useState } from 'react';
import { AuthModal } from './components/AuthModal';
import { TopBar, type AppRoute } from './components/TopBar';
import { AdminPage } from './pages/AdminPage';
import { ChatPage } from './pages/ChatPage';
import { CorpusPage } from './pages/CorpusPage';
import { HomePage } from './pages/HomePage';
import { dgeccApi, setAuthToken } from './services/api';
import type { AuthUser } from './types/dgecc';

export default function App() {
  const [route, setRoute] = useState<AppRoute>('home');
  const [user, setUser] = useState<AuthUser | null>(null);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authDefaultTab, setAuthDefaultTab] = useState<'login' | 'register' | 'bootstrap'>('login');
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);

  const handleOpenAuth = (tab: 'login' | 'register' | 'bootstrap' = 'login') => {
    setAuthDefaultTab(tab);
    setAuthModalOpen(true);
  };

  const handleLogout = async () => {
    try {
      await dgeccApi.logout();
    } catch {
      // Ignore network logout error
    }
    setAuthToken(null);
    setUser(null);
  };

  const handleStartQuestion = (question: string) => {
    setPendingPrompt(question);
    setRoute('chat');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F8F7F4] text-[#111827]">
      <TopBar
        currentRoute={route}
        onNavigate={setRoute}
        user={user}
        onOpenAuth={() => handleOpenAuth('login')}
        onLogout={handleLogout}
      />

      <div className="flex-1">
        {route === 'home' && (
          <HomePage onNavigate={setRoute} onStartQuestion={handleStartQuestion} />
        )}
        {route === 'chat' && (
          <ChatPage
            initialPrompt={pendingPrompt}
            onConsumedPrompt={() => setPendingPrompt(null)}
          />
        )}
        {route === 'corpus' && <CorpusPage />}
        {route === 'admin' && (
          <AdminPage
            user={user}
            onOpenAuth={(mode) => handleOpenAuth(mode)}
          />
        )}
      </div>

      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onAuthenticated={(authUser) => setUser(authUser)}
        defaultTab={authDefaultTab}
      />
    </div>
  );
}
