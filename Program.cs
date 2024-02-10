using ThreeOutputsNeuralNetwork;

string[] symptoms = { "Кашель", "Насмарк", "Головний бiль", "Температура", "Бiль в животi", "Провали в пам`ятi", "Проблеми зi сном" };

double[][] inputs =
{
               new double[] { 3, 3, 3, 3, 0, 0, 0 }, //хворий грипом
               new double[] { 3, 1, 2, 2, 0, 0, 0 }, //хворий грипом
               new double[] { 1, 0, 0, 1, 0, 0, 0 }, //хворий грипом
               new double[] { 0, 0, 0, 0, 3, 3, 3 }, //хворий пневмонією
               new double[] { 0, 0, 0, 1, 2, 0, 2 }, //хворий пневмонією
               new double[] { 0, 0, 0, 2, 2, 2, 0 }, //нхворий пневмонією
               new double[] { 0, 0, 0, 0, 0, 0, 0 }, //не хворий
               
            };

// Очікувані результати
double[] results1 = { 1, 1, 1, 0, 0, 0, 0 };
double[] results2 = { 0, 0, 0, 1, 1, 1, 0 };
double[] results3 = { 0, 0, 0, 0, 0, 0, 1 };

// Створення двох нейронів які приймають входи, та одного нейрона який видає результат
Neuron hiddenNeuron1 = new Neuron(7);
Neuron hiddenNeuron2 = new Neuron(7);
Neuron hiddenNeuron3 = new Neuron(7);
Neuron hiddenNeuron4 = new Neuron(7);
Neuron hiddenNeuron5 = new Neuron(7);

Neuron outputNeuron1 = new Neuron(5);
Neuron outputNeuron2 = new Neuron(5);
Neuron outputNeuron3 = new Neuron(5);

// Нормалізація входів
for (int i = 0; i < inputs.Length; i++)
{
    inputs[i] = Normalizator.Normalize(inputs[i]);
}

int epoch = 0;
bool result1 = false;
bool result2 = false;
bool result3 = false;
int trueResultsCount = 0;

while (trueResultsCount < (results1.Length + results2.Length + results3.Length))
{
    epoch++;
    for (int i = 0; i < results1.Length; i++)
    {
        // Передача входів на кожен нейрон

        hiddenNeuron1.SetInputs(inputs[i]);
        hiddenNeuron2.SetInputs(inputs[i]);
        hiddenNeuron3.SetInputs(inputs[i]);
        hiddenNeuron4.SetInputs(inputs[i]);
        hiddenNeuron5.SetInputs(inputs[i]);

        outputNeuron1.SetInputs(new double[] { 
            hiddenNeuron1.CalculateOutput(), 
            hiddenNeuron2.CalculateOutput(), 
            hiddenNeuron3.CalculateOutput(), 
            hiddenNeuron4.CalculateOutput(),
            hiddenNeuron5.CalculateOutput()});

        outputNeuron2.SetInputs(new double[] {
            hiddenNeuron1.CalculateOutput(),
            hiddenNeuron2.CalculateOutput(),
            hiddenNeuron3.CalculateOutput(),
            hiddenNeuron4.CalculateOutput(),
            hiddenNeuron5.CalculateOutput()});

        outputNeuron3.SetInputs(new double[] {
            hiddenNeuron1.CalculateOutput(),
            hiddenNeuron2.CalculateOutput(),
            hiddenNeuron3.CalculateOutput(),
            hiddenNeuron4.CalculateOutput(),
            hiddenNeuron5.CalculateOutput()});

        // Перевірка на вірний результат
        result1 = outputNeuron1.CheckResult(results1, i);
        result2 = outputNeuron2.CheckResult(results2, i);
        result3 = outputNeuron3.CheckResult(results3, i);

        if (result1 == true && result2 == true && result3 == true)
        {
            trueResultsCount+= 3;
        }
        else
        {
            trueResultsCount = 0;
        }

        Console.WriteLine($"for input {i} (1 output neuron) = {outputNeuron1.CalculateOutput()} - {result1}, epoch: {epoch}");
        Console.WriteLine($"for input {i} (2 output neuron) = {outputNeuron2.CalculateOutput()} - {result2}, epoch: {epoch}");
        Console.WriteLine($"for input {i} (3 output neuron) = {outputNeuron3.CalculateOutput()} - {result3}, epoch: {epoch}");
        Console.WriteLine();

        // Зміна ваг вихідного нейрона відносно його нейронної похибки
        outputNeuron1.SetNeuronError(Sigmoid.CalculateDerivative(outputNeuron1.CalculateOutput()) * (results1[i] - outputNeuron1.CalculateOutput()));
        outputNeuron1.UpdateWeights();

        outputNeuron2.SetNeuronError(Sigmoid.CalculateDerivative(outputNeuron2.CalculateOutput()) * (results2[i] - outputNeuron2.CalculateOutput()));
        outputNeuron2.UpdateWeights();

        outputNeuron3.SetNeuronError(Sigmoid.CalculateDerivative(outputNeuron3.CalculateOutput()) * (results3[i] - outputNeuron3.CalculateOutput()));
        outputNeuron3.UpdateWeights();

        // Зміна ваг вхідних нейронів, відносно їхньої нейронної похибки
        hiddenNeuron1.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron1.CalculateOutput()) * outputNeuron1.NeuronError * outputNeuron1.Weights[0]);
        hiddenNeuron2.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron1.NeuronError * outputNeuron1.Weights[1]);
        hiddenNeuron3.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron1.NeuronError * outputNeuron1.Weights[2]);
        hiddenNeuron4.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron1.NeuronError * outputNeuron1.Weights[3]);
        hiddenNeuron5.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron1.NeuronError * outputNeuron1.Weights[4]);

        hiddenNeuron1.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron1.CalculateOutput()) * outputNeuron2.NeuronError * outputNeuron2.Weights[0]);
        hiddenNeuron2.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron2.NeuronError * outputNeuron2.Weights[1]);
        hiddenNeuron3.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron2.NeuronError * outputNeuron2.Weights[2]);
        hiddenNeuron4.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron2.NeuronError * outputNeuron2.Weights[3]);
        hiddenNeuron5.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron2.NeuronError * outputNeuron2.Weights[4]);

        hiddenNeuron1.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron1.CalculateOutput()) * outputNeuron3.NeuronError * outputNeuron3.Weights[0]);
        hiddenNeuron2.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron3.NeuronError * outputNeuron3.Weights[1]);
        hiddenNeuron3.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron3.NeuronError * outputNeuron3.Weights[2]);
        hiddenNeuron4.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron3.NeuronError * outputNeuron3.Weights[3]);
        hiddenNeuron5.SetNeuronError(Sigmoid.CalculateDerivative(hiddenNeuron2.CalculateOutput()) * outputNeuron3.NeuronError * outputNeuron3.Weights[4]);

        hiddenNeuron1.UpdateWeights();
        hiddenNeuron2.UpdateWeights();
        hiddenNeuron3.UpdateWeights();
        hiddenNeuron4.UpdateWeights();
        hiddenNeuron5.UpdateWeights();
    }

    Console.WriteLine();
}

var healthChecker = new HealthChecker(
    7, 
    symptoms, 
    hiddenNeuron1, 
    hiddenNeuron2, 
    hiddenNeuron3, 
    hiddenNeuron4, 
    hiddenNeuron5, 
    outputNeuron1,
    outputNeuron2,
    outputNeuron3
    );

healthChecker.CheckHealth();
