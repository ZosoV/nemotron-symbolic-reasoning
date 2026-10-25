import InspectorView from '../components/InspectorView';
import { RESULTS_COLUMNS } from '../lib/schema';

export default function ResultsPage() {
  return <InspectorView dir="results" schema={RESULTS_COLUMNS} />;
}
