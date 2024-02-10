namespace ThreeOutputsNeuralNetwork
{
    class Normalizator
    {
        private static double _min;
        private static double _max;

        public static double[] NormalizeOnly(double[] inputs)
        {
            double[] normalizedInputs = new double[inputs.Length];

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] != 0)
                    normalizedInputs[i] = (inputs[i] - _min) / (_max - _min);
            }

            return normalizedInputs;
        }

        public static double[] Normalize(double[] inputs)
        {
            double[] normalizedInputs = new double[inputs.Length];

            double min = inputs[0];
            double max = inputs[0];

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] < min)
                {
                    min = inputs[i];
                }
                if (inputs[i] > max)
                {
                    max = inputs[i];
                }
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] != 0)
                normalizedInputs[i] = (inputs[i] - min) / (max - min);
            }

            _max = max;
            _min = min;

            return normalizedInputs;
        }
    }
}
