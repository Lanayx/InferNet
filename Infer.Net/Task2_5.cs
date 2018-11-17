using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using System;
using System.Collections.Generic;
using System.Text;

namespace Infer.Net
{
    public static class Task2_5
    {
        public static void Run()
        {
            var skills = new Range(7);
            var skillsVars = Variable.Array<bool>(skills);
            skillsVars[skills] = Variable.Bernoulli(0.5).ForEach(skills);


        }
    }
}
