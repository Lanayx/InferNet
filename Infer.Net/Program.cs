using Microsoft.ML.Probabilistic.Models;
using System;

namespace Infer.Net
{
    class Program
    {
        static void Main(string[] args)
        {
            Variable<bool> knowsCsharp = Variable.Bernoulli(0.5);
            Variable<bool> knowsSql = Variable.Bernoulli(0.5);
            Variable<bool> knowsBoth = knowsCsharp & knowsSql;
            // sprinkler
            Variable<bool> firstQuestion = Variable.New<bool>();
            using (Variable.If(knowsCsharp)) firstQuestion.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(knowsCsharp)) firstQuestion.SetTo(Variable.Bernoulli(0.2));
            // rain
            Variable<bool> secondQuestion = Variable.New<bool>();
            using (Variable.If(knowsSql)) secondQuestion.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(knowsSql)) secondQuestion.SetTo(Variable.Bernoulli(0.2));
            // wet grass
            Variable<bool> thirdQuestion = Variable.New<bool>();
            using (Variable.If(knowsBoth)) thirdQuestion.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(knowsBoth)) thirdQuestion.SetTo(Variable.Bernoulli(0.2));

            // Observations and inference
            firstQuestion.ObservedValue = false;
            secondQuestion.ObservedValue = false;
            thirdQuestion.ObservedValue = true;
            InferenceEngine ie = new InferenceEngine();
            ie.ShowProgress = false;
            Console.WriteLine("P(csharp)=" + ie.Infer(knowsCsharp));
            Console.WriteLine("P(sql)=" + ie.Infer(knowsSql));
        }
    }
}
