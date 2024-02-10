using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ThreeOutputsNeuralNetwork
{
    public class HealthChecker
    {
        private double[] _inputs;
        private string[] _symptoms;

        private Neuron _hiddenNeuron1;
        private Neuron _hiddenNeuron2;
        private Neuron _hiddenNeuron3;
        private Neuron _hiddenNeuron4;
        private Neuron _hiddenNeuron5;

        private Neuron _outputNeuron1;
        private Neuron _outputNeuron2;
        private Neuron _outputNeuron3;

        public HealthChecker(
            int inputsCount, string[] symptoms, 
            Neuron hiddenNeuron1, Neuron hiddenNeuron2, Neuron hiddenNeuron3, Neuron hiddenNeuron4, Neuron hiddenNeuron5, 
            Neuron outputNeuron1, Neuron outputNeuron2, Neuron outputNeuron3)
        {
            _inputs = new double[inputsCount];
            _symptoms = symptoms;

            _hiddenNeuron1 = hiddenNeuron1;
            _hiddenNeuron2 = hiddenNeuron2;
            _hiddenNeuron3 = hiddenNeuron3;
            _hiddenNeuron4 = hiddenNeuron4;
            _hiddenNeuron5 = hiddenNeuron5;

            _outputNeuron1 = outputNeuron1;
            _outputNeuron2 = outputNeuron2;
            _outputNeuron3 = outputNeuron3;
        }

        private void SetSymptoms()
        {
            Console.WriteLine("Надайте симптоми щоб перевiрити чи хворi ви грипом:");
            Console.WriteLine();

            for (int i = 0; i < _inputs.Length; i++)
            {
                Console.Write($"{_symptoms[i]} (форма проявлення симптому: 0 - 3): ");
                _inputs[i] = double.Parse(Console.ReadLine());
            }

            _inputs = Normalizator.NormalizeOnly(_inputs);
        }

        public void CheckHealth()
        {
            string input = String.Empty;

            while (input != "n")
            {
                SetSymptoms();

                _hiddenNeuron1.SetInputs(_inputs);
                _hiddenNeuron2.SetInputs(_inputs);
                _hiddenNeuron3.SetInputs(_inputs);
                _hiddenNeuron4.SetInputs(_inputs);
                _hiddenNeuron5.SetInputs(_inputs);

                _outputNeuron1.SetInputs(new double[] { 
                    _hiddenNeuron1.CalculateOutput(), 
                    _hiddenNeuron2.CalculateOutput(), 
                    _hiddenNeuron3.CalculateOutput(), 
                    _hiddenNeuron4.CalculateOutput(), 
                    _hiddenNeuron5.CalculateOutput(), 
                });

                _outputNeuron2.SetInputs(new double[] {
                    _hiddenNeuron1.CalculateOutput(),
                    _hiddenNeuron2.CalculateOutput(),
                    _hiddenNeuron3.CalculateOutput(),
                    _hiddenNeuron4.CalculateOutput(),
                    _hiddenNeuron5.CalculateOutput(),
                });

                _outputNeuron3.SetInputs(new double[] {
                    _hiddenNeuron1.CalculateOutput(),
                    _hiddenNeuron2.CalculateOutput(),
                    _hiddenNeuron3.CalculateOutput(),
                    _hiddenNeuron4.CalculateOutput(),
                    _hiddenNeuron5.CalculateOutput(),
                });

                double result1 = _outputNeuron1.CalculateOutput();
                double result2 = _outputNeuron2.CalculateOutput();
                double result3 = _outputNeuron3.CalculateOutput();

                Console.WriteLine();

                if (result1 >= 0.9d)
                {
                    Console.WriteLine($"Ви хворi на грип на 100%");
                }
                else if (result1 < 0.9d && result1 >= 0.1d)
                {
                    Console.WriteLine($"Ви хворi на грип на {result1 * 100:00}%");
                }
                else if (result1 < 0.1d)
                {
                    Console.WriteLine($"Ви хворi на грип на {result1 * 100:0}%");
                }

                if (result2 >= 0.9d)
                {
                    Console.WriteLine($"Ви хворi на пневмонiю на 100%");
                }
                else if (result2 < 0.9d && result2 >= 0.1d)
                {
                    Console.WriteLine($"Ви хворi на пневмонiю на {result2 * 100:00}%");
                }
                else if (result2 < 0.1d)
                {
                    Console.WriteLine($"Ви хворi на пневмонiю на {result2 * 100:0}%");
                }

                if (result3 >= 0.9d)
                {
                    Console.WriteLine($"Ви здоровi на 100%");
                }
                else if (result3 < 0.9d && result3 >= 0.1d)
                {
                    Console.WriteLine($"Ви здоровi на {result3 * 100:00}%");
                }
                else if (result3 < 0.1d)
                {
                    Console.WriteLine($"Ви здоровi на {result3 * 100:0}%");
                }

                Console.WriteLine();
                Console.Write("Continue? (y/n): ");
                input = Console.ReadLine();
                Console.WriteLine();
            }
        }
    }
}
