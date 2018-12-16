using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using System;
using System.Collections.Generic;
using System.Text;

namespace Infer.Net
{
    public static class Task3_3
    {
        public static void Run()
        {
            var jSkill = Variable.GaussianFromMeanAndVariance(120,1600);
            var fSkill = Variable.GaussianFromMeanAndVariance(100, 25);

            var jPerf = Variable.GaussianFromMeanAndVariance(jSkill, 25);
            var fPerf = Variable.GaussianFromMeanAndVariance(fSkill, 25);

            var jWins = jPerf > fPerf;

            //Console.WriteLine("Jill wins");
            //var ie = new InferenceEngine();
            //jWins.ObservedValue = true;
            //Console.WriteLine("P(jSkill)=" + ie.Infer(jSkill));
            //Console.WriteLine("P(fSkill)=" + ie.Infer(fSkill));
            Console.WriteLine("Fred wins");
            var ie = new InferenceEngine();
            jWins.ObservedValue = false;
            Console.WriteLine("P(fSkill)=" + ie.Infer<Gaussian>(fSkill));
            Console.WriteLine("P(jSkill)=" + ie.Infer<Gaussian>(jSkill));
        }
    }
}
