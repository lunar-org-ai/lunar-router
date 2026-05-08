import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { RouterProvider } from '@tanstack/react-router';
import { router } from './router';
import './styles.css';

router.subscribe('onResolved', () => {
  document.querySelector('.main')?.scrollTo(0, 0);
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
