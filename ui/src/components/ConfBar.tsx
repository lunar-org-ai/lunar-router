export const ConfBar = ({ value, max = 5 }: { value: number; max?: number }) => (
  <span className="conf-bar" title={`${value}/${max}`}>
    {Array.from({ length: max }, (_, i) => (
      <i key={i} className={i < value ? 'on' : ''} />
    ))}
  </span>
);
