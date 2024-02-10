using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ThreeOutputsNeuralNetwork
{
    public class Neuron
    {
        private double[] _inputs;
        private double[] _weights;
        private double _error;
        private double _biasWeight;
        private double _learningRate = 1;

        public Neuron(int inputsCount)
        {
            Random random = new Random();
            _weights = new double[inputsCount];
            _inputs = new double[inputsCount];

            for (int i = 0; i < inputsCount; i++)
            {
                _weights[i] = random.NextDouble();
            }

            _biasWeight = random.NextDouble();
        }

        public double[] Weights { get { return _weights; } }
        public double NeuronError { get { return _error; } }

        public double CalculateOutput()
        {
            double sum = 0;

            for (int i = 0; i < _inputs.Length; i++)
            {
                sum += _weights[i] * _inputs[i];
            }

            sum += _biasWeight;

            return Sigmoid.CalculateOutput(sum);
        }

        public void SetNeuronError(double error)
        {
            _error = error;
        }

        public void SetInputs(double[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                _inputs[i] = inputs[i];
            }
        }

        public void UpdateWeights()
        {
            for (int i = 0; i < _inputs.Length; i++)
            {
                _weights[i] += _learningRate * _error * _inputs[i];
            }

            _biasWeight += _learningRate * _error;
        }

        public bool CheckResult(double[] results, int iteration)
        {
            if (results[iteration] == 1)
            {
                if (CalculateOutput() > 0.9d)
                {
                    return true;
                }
            }
            else if (results[iteration] == 0)
            {
                if (CalculateOutput() < 0.1d)
                {
                    return true;
                }
            }

            return false;
        }
    }
}
