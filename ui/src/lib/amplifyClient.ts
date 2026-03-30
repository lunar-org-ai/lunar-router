import { generateClient } from 'aws-amplify/data';
import type { Schema } from '../../amplify/data/resource';

let _client: ReturnType<typeof generateClient<Schema>> | null = null;

/**
 * Lazily initializes the Amplify Data client.
 * Must be called after Amplify.configure() has run (i.e., not at module level).
 */
export function getAmplifyClient() {
  if (!_client) _client = generateClient<Schema>();
  return _client;
}
