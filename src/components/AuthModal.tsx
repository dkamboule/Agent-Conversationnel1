import { useEffect, useState, type FormEvent } from 'react';
import { X } from 'lucide-react';
import { dgeccApi, setAuthToken } from '../services/api';
import type { AuthUser } from '../types/dgecc';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAuthenticated: (user: AuthUser) => void;
  defaultTab?: 'login' | 'register' | 'bootstrap';
}

export function AuthModal({
  isOpen,
  onClose,
  onAuthenticated,
  defaultTab = 'login',
}: AuthModalProps) {
  const [mode, setMode] = useState<'login' | 'register' | 'bootstrap'>(defaultTab);
  const [needsAdminSetup, setNeedsAdminSetup] = useState(false);
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [service, setService] = useState('Direction Générale DGECC');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setError(null);
    dgeccApi
      .getBootstrapStatus()
      .then((res) => {
        setNeedsAdminSetup(res.needsAdminSetup);
        if (res.needsAdminSetup && defaultTab === 'bootstrap') {
          setMode('bootstrap');
        } else {
          setMode(defaultTab === 'bootstrap' ? 'login' : defaultTab);
        }
      })
      .catch(() => {
        // Ignore transient status failure
      });
  }, [isOpen, defaultTab]);

  if (!isOpen) return null;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      let res: { token: string; user: AuthUser };
      if (mode === 'bootstrap') {
        res = await dgeccApi.bootstrapAdmin({ email, fullName, password, service });
      } else if (mode === 'register') {
        res = await dgeccApi.register({ email, fullName, password });
      } else {
        res = await dgeccApi.login({ email, password });
      }
      setAuthToken(res.token);
      onAuthenticated(res.user);
      setPassword('');
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Échec de l’authentification.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="auth-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4"
    >
      <div className="w-full max-w-md bg-white border border-[#E5E2DC] rounded-lg shadow-xl p-6">
        <div className="flex items-start justify-between gap-4 pb-4 border-b border-[#E5E2DC]">
          <div>
            <h2 id="auth-modal-title" className="font-serif text-xl font-semibold text-[#111827]">
              {mode === 'bootstrap'
                ? 'Initialisation du compte Administrateur'
                : mode === 'register'
                  ? 'Créer un compte usager'
                  : 'Connexion à l’espace DGECC'}
            </h2>
            <p className="text-xs text-[#6B7280] mt-1">
              {mode === 'bootstrap'
                ? 'Premier démarrage : définissez les identifiants de l’administrateur principal.'
                : 'Accédez à votre historique de demandes ou à la console d’administration.'}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Fermer la fenêtre de connexion"
            className="p-1.5 text-[#6B7280] hover:text-[#111827] rounded-md cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Sélecteur de mode */}
        <div className="flex items-center gap-1 p-1 bg-[#F3F1EC] rounded-md mt-4">
          <button
            type="button"
            onClick={() => {
              setMode('login');
              setError(null);
            }}
            className={`flex-1 py-1.5 px-3 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap ${
              mode === 'login' ? 'bg-white text-[#111827] shadow-xs' : 'text-[#4B5563] hover:text-[#111827]'
            }`}
          >
            Se connecter
          </button>
          <button
            type="button"
            onClick={() => {
              setMode('register');
              setError(null);
            }}
            className={`flex-1 py-1.5 px-3 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap ${
              mode === 'register'
                ? 'bg-white text-[#111827] shadow-xs'
                : 'text-[#4B5563] hover:text-[#111827]'
            }`}
          >
            Compte usager
          </button>
          {needsAdminSetup && (
            <button
              type="button"
              onClick={() => {
                setMode('bootstrap');
                setError(null);
              }}
              className={`flex-1 py-1.5 px-3 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap ${
                mode === 'bootstrap'
                  ? 'bg-white text-[#1E3A8A] shadow-xs'
                  : 'text-[#4B5563] hover:text-[#111827]'
              }`}
            >
              Config. Admin
            </button>
          )}
        </div>

        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          {(mode === 'register' || mode === 'bootstrap') && (
            <div>
              <label className="block text-xs font-medium text-[#374151] mb-1">
                Nom complet
              </label>
              <input
                type="text"
                required
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Ex. Marie Diallo"
                className="w-full px-3 py-2 text-sm border border-[#D6D3CD] rounded-md focus:outline-2 focus:outline-[#1E3A8A]"
              />
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-[#374151] mb-1">
              Adresse courriel institutionnelle ou personnelle
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="nom@exemple.fr"
              className="w-full px-3 py-2 text-sm border border-[#D6D3CD] rounded-md focus:outline-2 focus:outline-[#1E3A8A]"
            />
          </div>

          {mode === 'bootstrap' && (
            <div>
              <label className="block text-xs font-medium text-[#374151] mb-1">
                Service de rattachement
              </label>
              <input
                type="text"
                value={service}
                onChange={(e) => setService(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-[#D6D3CD] rounded-md focus:outline-2 focus:outline-[#1E3A8A]"
              />
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-[#374151] mb-1">
              Mot de passe {mode === 'bootstrap' ? '(8 caractères min.)' : '(6 caractères min.)'}
            </label>
            <input
              type="password"
              required
              minLength={mode === 'bootstrap' ? 8 : 6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-[#D6D3CD] rounded-md focus:outline-2 focus:outline-[#1E3A8A]"
            />
          </div>

          {error && (
            <div
              role="alert"
              className="p-3 text-xs text-[#991B1B] bg-[#FEF2F2] border border-[#FECACA] rounded-md"
            >
              {error}
            </div>
          )}

          <div className="pt-2 flex items-center justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-[#4B5563] hover:text-[#111827] cursor-pointer"
            >
              Annuler
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-5 py-2 text-sm font-medium text-white bg-[#1E3A8A] hover:bg-[#172E6E] disabled:opacity-50 rounded-md cursor-pointer"
            >
              {loading
                ? 'Traitement...'
                : mode === 'bootstrap'
                  ? 'Créer l’administrateur'
                  : mode === 'register'
                    ? 'Créer le compte'
                    : 'Se connecter'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
