interface RadioGroupProps<T extends string> {
  title: string;
  name: string;
  options: Array<{
    label: string;
    value: T;
    disabled?: boolean;
  }>;
  selectedValue: T;
  onChange: (value: T) => void;
}

export default function RadioGroup<T extends string>({
  title,
  name,
  options,
  selectedValue,
  onChange,
}: RadioGroupProps<T>) {
  return (
    <form className="bg-white p-4 rounded-lg shadow flex-1">
      <h2 className="text-xl font-bold text-gray-800 mb-3">{title}</h2>
      <div className="space-y-2">
        {options.map(({ label, value, disabled }) => (
          <label
            key={value}
            className={`flex items-center space-x-3 py-1 ${
              disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'
            }`}
          >
            <input
              type="radio"
              name={name}
              value={value}
              checked={selectedValue === value}
              onChange={(e) => onChange(e.target.value as T)}
              disabled={disabled}
              className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500 disabled:cursor-not-allowed"
            />
            <span className={`${disabled ? 'text-gray-400' : 'text-gray-700'}`}>
              {label}
            </span>
          </label>
        ))}
      </div>
    </form>
  );
}