import type { AuthUser } from '../types/dgecc';

export type AppRoute = 'home' | 'chat' | 'corpus' | 'admin';

interface TopBarProps {
  currentRoute: AppRoute;
  onNavigate: (route: AppRoute) => void;
  user: AuthUser | null;
  onOpenAuth: () => void;
  onLogout: () => void;
}

export function TopBar({
  currentRoute,
  onNavigate,
  user,
  onOpenAuth,
  onLogout,
}: TopBarProps) {
  const navItems: Array<{ id: AppRoute; label: string }> = [
    { id: 'home', label: 'Accueil' },
    { id: 'chat', label: 'Assistant' },
    { id: 'corpus', label: 'Corpus officiel' },
    { id: 'admin', label: 'Administration' },
  ];

  return (
    <header className="sticky top-0 z-30 bg-[#F8F7F4]/95 backdrop-blur-sm border-b border-[#E5E2DC]">
      <div className="max-w-[1360px] mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-8">
        {/* Zone 1 : Marque textuelle sur une ligne */}
        <button
          type="button"
          onClick={() => onNavigate('home')}
          className="font-serif text-lg sm:text-xl font-semibold tracking-tight text-[#111827] whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-[#1E3A8A] rounded-sm cursor-pointer"
        >
          DGECC Orientation
        </button>

        {/* Zone 2 : Liens de navigation concis sur une ligne */}
        <nav
          aria-label="Navigation principale"
          className="hidden md:flex items-center gap-7 text-sm font-medium"
        >
          {navItems.map((item) => {
            const active = currentRoute === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavigate(item.id)}
                className={`py-1 whitespace-nowrap shrink-0 transition-colors duration-150 border-b-2 cursor-pointer focus-visible:outline-2 focus-visible:outline-[#1E3A8A] ${
                  active
                    ? 'text-[#1E3A8A] border-[#1E3A8A]'
                    : 'text-[#4B5563] border-transparent hover:text-[#111827] hover:border-[#D1D5DB]'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Zone 3 : Action principale unique */}
        <div className="flex items-center shrink-0">
          {user ? (
            <button
              type="button"
              onClick={onLogout}
              title={`Connecté en tant que ${user.email} (${user.role})`}
              className="px-4 py-2 text-xs sm:text-sm font-medium text-[#111827] bg-white border border-[#D6D3CD] rounded-md hover:bg-[#F3F1EC] transition-colors duration-150 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-[#1E3A8A]"
            >
              Déconnexion ({user.fullName.split(' ')[0]})
            </button>
          ) : (
            <button
              type="button"
              onClick={onOpenAuth}
              className="px-4 py-2 text-xs sm:text-sm font-medium text-white bg-[#1E3A8A] rounded-md hover:bg-[#172E6E] transition-colors duration-150 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1E3A8A]"
            >
              Connexion
            </button>
          )}
        </div>
      </div>

      {/* Navigation mobile compacte */}
      <nav
        aria-label="Navigation mobile"
        className="flex md:hidden items-center justify-around border-t border-[#E5E2DC] px-2 py-2 bg-[#F8F7F4]"
      >
        {navItems.map((item) => {
          const active = currentRoute === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onNavigate(item.id)}
              className={`px-2 py-1 text-xs font-medium whitespace-nowrap shrink-0 ${
                active ? 'text-[#1E3A8A] underline underline-offset-4' : 'text-[#4B5563]'
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
