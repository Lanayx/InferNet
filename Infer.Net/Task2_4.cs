using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using System;
using System.Collections.Generic;
using System.Text;

namespace Infer.Net
{
    public static class Task2_4
    {
        public static void Run()
        {
            Variable<bool> knowsCsharp = Variable.Bernoulli(0.5).Attrib(new TraceMessages());
            Variable<bool> knowsSql = Variable.Bernoulli(0.5).Attrib(new TraceMessages());
            Variable<bool> knowsBoth = knowsCsharp & knowsSql;

            Variable<bool> firstQuestion = Variable.New<bool>();
            using (Variable.If(knowsCsharp)) firstQuestion.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(knowsCsharp)) firstQuestion.SetTo(Variable.Bernoulli(0.2));

            Variable<bool> secondQuestion = Variable.New<bool>();
            using (Variable.If(knowsSql)) secondQuestion.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(knowsSql)) secondQuestion.SetTo(Variable.Bernoulli(0.2));

            Variable<bool> thirdQuestion = Variable.New<bool>();
            using (Variable.If(knowsBoth)) thirdQuestion.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(knowsBoth)) thirdQuestion.SetTo(Variable.Bernoulli(0.2));

            Variable<bool> fourthQuestion = Variable.New<bool>();
            using (Variable.If(knowsBoth)) fourthQuestion.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(knowsBoth)) fourthQuestion.SetTo(Variable.Bernoulli(0.2));

            // Observations and inference
            firstQuestion.ObservedValue = true;
            secondQuestion.ObservedValue = false;
            thirdQuestion.ObservedValue = false;
            fourthQuestion.ObservedValue = false;
            InferenceEngine ie = new InferenceEngine();
            ie.ShowTimings = true;
            ie.ShowProgress = true;
            ie.ShowSchedule = true;
            ie.ShowFactorGraph = true;
            Console.WriteLine("P(csharp)=" + ie.Infer(knowsCsharp));
            Console.WriteLine("P(sql)=" + ie.Infer(knowsSql));
        }
    }
}
