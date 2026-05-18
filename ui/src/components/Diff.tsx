export type DiffLineKind = 'add' | 'del' | 'ctx';
export type DiffLine = [DiffLineKind, string];

export const Diff = ({
  before,
  after,
  fileLabel,
}: {
  before: DiffLine[];
  after: DiffLine[];
  fileLabel: string;
}) => (
  <div className="diff">
    <div className="diff-head">
      <span>{fileLabel}</span>
      <span className="dim mono" style={{ fontSize: 11 }}>
        −{before.filter((l) => l[0] === 'del').length} +{after.filter((l) => l[0] === 'add').length}
      </span>
    </div>
    <div className="diff-body">
      {after.map(([t, line], i) => (
        <div key={i} className={`diff-line ${t === 'add' ? 'add' : t === 'del' ? 'del' : ''}`}>
          <span className="gut">{t === 'add' || t === 'del' ? '' : i + 1}</span>
          <span>{line || ' '}</span>
        </div>
      ))}
    </div>
  </div>
);
