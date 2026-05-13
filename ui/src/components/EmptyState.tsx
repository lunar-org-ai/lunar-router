/**
 * EmptyState — shared "nothing here yet" surface, lifted from the
 * design's day-zero treatment so the technical tabs and Evolution stay
 * visually consistent.
 *
 * Two flavors:
 *   - **No-data** (kind="day-zero" or omitted): big icon + friendly
 *     copy explaining what this surface fills with + an optional CTA.
 *     Used the first time the operator opens Traces/Datasets/Versions
 *     for a fresh agent.
 *   - **Filtered**: tight one-liner + "Clear filters" button. The
 *     design uses `.empty-state` for this case so we keep it.
 */

import { type ReactNode } from 'react';

import { Icon, type IconName } from './Icon';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface EmptyStateProps {
  title: string;
  description?: ReactNode;
  icon?: IconName;
  /** When provided, renders a primary CTA. */
  cta?: {
    label: string;
    onClick?: () => void;
    href?: string;
  };
  /** Filter-clear button (small, ghost). Shown below the CTA. */
  secondary?: {
    label: string;
    onClick: () => void;
  };
}

export const EmptyState = ({
  title,
  description,
  icon,
  cta,
  secondary,
}: EmptyStateProps) => (
  <Card className="day-zero-card">
    {icon && (
      <div className="day-zero-icon" aria-hidden="true">
        <Icon name={icon} size={28} />
      </div>
    )}
    <div className="day-zero-title">{title}</div>
    {description && <div className="day-zero-desc dim">{description}</div>}
    {(cta || secondary) && (
      <div className="day-zero-actions">
        {cta &&
          (cta.href ? (
            <Button asChild>
              <a href={cta.href}>
                {cta.label} <Icon name="chevron" size={14} />
              </a>
            </Button>
          ) : (
            <Button onClick={cta.onClick}>
              {cta.label} <Icon name="chevron" size={14} />
            </Button>
          ))}
        {secondary && (
          <Button variant="ghost" size="sm" onClick={secondary.onClick}>
            {secondary.label}
          </Button>
        )}
      </div>
    )}
  </Card>
);
