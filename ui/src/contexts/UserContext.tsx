import type { ReactNode } from 'react';
import {
  createContext,
  useContext,
  useEffect,
  useState,
  useRef,
  useMemo,
  useCallback,
} from 'react';
import type { Schema } from '../../amplify/data/resource';
import type { KeyResponse } from '../services/keyService';
import { keyService } from '../services/keyService';
import { useProfileService, type Profile } from '../services/profileService';
import { useUserService } from '../hooks/useUserService';

type User = NonNullable<Schema['Users']['type']>;
type ProfileCompat = { tenant_id?: string; tenant?: { id?: string } };

interface UserContextType {
  user: User | null;
  idToken: string | null;
  accessToken: string | null;
  tenantId: string | null;
  loading: boolean;
  error: string | null;
  refetchUser: () => Promise<void>;
  apiKey: KeyResponse | null;
  profile: Profile | null;
  updateProfile: (updates: { onboarding_step?: number; is_new_user?: boolean }) => Promise<void>;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

interface UserProviderProps {
  children: ReactNode;
}

const extractTenantId = (profile: ProfileCompat | null): string | null =>
  profile?.tenant_id || profile?.tenant?.id || null;

export const UserProvider = ({ children }: UserProviderProps) => {
  const { user, idToken, accessToken, loading, error, fetchUser } = useUserService();
  const { profile: userProfile, fetchProfile, updateProfile: patchProfile } = useProfileService();

  const [isInitializing, setIsInitializing] = useState(true);
  const [tenantId, setTenantId] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<KeyResponse | null>(null);
  const hasFetchedRef = useRef(false);

  // load profile and extract tenantId
  const loadProfile = useCallback(
    async (token: string): Promise<string | null> => {
      try {
        const profile = await fetchProfile(token);
        const extractedTenantId = extractTenantId(profile);

        if (extractedTenantId) {
          setTenantId(extractedTenantId);
          return extractedTenantId;
        }

        console.warn('[UserContext] Could not extract tenantId from profile:', profile);
        return null;
      } catch (err) {
        console.warn('[UserContext] Error fetching profile:', err);
        return null;
      }
    },
    [fetchProfile]
  );

  const ensureDefaultApiKey = useCallback(async (token: string) => {
    try {
      const { createKey } = keyService(token);
      const defaultKey = await createKey('psk-first-key-psk');
      setApiKey(defaultKey);
    } catch (err) {
      console.warn('[UserContext] Error ensuring default key:', err);
    }
  }, []);

  // load user data on mount
  useEffect(() => {
    if (hasFetchedRef.current) return;
    hasFetchedRef.current = true;

    let isMounted = true;

    const initialize = async () => {
      try {
        const { idToken, accessToken } = await fetchUser();
        const profileToken = idToken || accessToken;
        const keyToken = accessToken || idToken;

        if (!isMounted) return;

        await Promise.all([
          profileToken ? loadProfile(profileToken) : Promise.resolve(null),
          keyToken ? ensureDefaultApiKey(keyToken) : Promise.resolve(),
        ]);
      } finally {
        if (isMounted) setIsInitializing(false);
      }
    };

    initialize();

    return () => {
      isMounted = false;
      hasFetchedRef.current = false;
    };
  }, [fetchUser, loadProfile, ensureDefaultApiKey]);

  const refetchUser = useCallback(async () => {
    const { idToken, accessToken } = await fetchUser();
    const profileToken = idToken || accessToken;
    if (profileToken) {
      await loadProfile(profileToken);
    }
  }, [fetchUser, loadProfile]);

  const updateProfile = useCallback(
    async (updates: { onboarding_step?: number; is_new_user?: boolean }) => {
      if (!accessToken) throw new Error('Not authenticated');
      await patchProfile(accessToken, updates);
    },
    [accessToken, patchProfile]
  );

  const value = useMemo(
    () => ({
      user,
      idToken,
      accessToken,
      tenantId,
      loading: loading || isInitializing,
      error,
      refetchUser,
      apiKey,
      profile: userProfile,
      updateProfile,
    }),
    [
      user,
      idToken,
      accessToken,
      tenantId,
      loading,
      isInitializing,
      error,
      apiKey,
      refetchUser,
      userProfile,
      updateProfile,
    ]
  );

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
};

const FALLBACK_USER_CONTEXT: UserContextType = {
  user: null,
  idToken: null,
  accessToken: null,
  tenantId: null,
  loading: false,
  error: null,
  refetchUser: async () => {},
  apiKey: null,
  profile: null,
  updateProfile: async () => {},
};

export const useUser = (): UserContextType => {
  const context = useContext(UserContext);
  return context ?? FALLBACK_USER_CONTEXT;
};
