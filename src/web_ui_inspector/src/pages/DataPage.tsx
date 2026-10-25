import InspectorView from '../components/InspectorView';
import { DATASET_COLUMNS } from '../lib/schema';

export default function DataPage() {
  return <InspectorView dir="data" schema={DATASET_COLUMNS} />;
}
