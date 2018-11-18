using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Infer.Net
{
    public static class Task2_5
    {
        public static T[][] TransposeRowsAndColumns<T>(this T[][] arr)
        {
            int rowCount = arr.Length;
            int columnCount = arr[0].Length;
            T[][] transposed = new T[columnCount][];
            if (rowCount == columnCount)
            {
                transposed = (T[][])arr.Clone();
                for (int i = 1; i < rowCount; i++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        T temp = transposed[i][j];
                        transposed[i][j] = transposed[j][i];
                        transposed[j][i] = temp;
                    }
                }
            }
            else
            {
                for (int column = 0; column < columnCount; column++)
                {
                    transposed[column] = new T[rowCount];
                    for (int row = 0; row < rowCount; row++)
                    {
                        transposed[column][row] = arr[row][column];
                    }
                }
            }
            return transposed;
        }

        public static bool[][] LoadQuestionsVsSkills()
        {
            var csv = File.ReadAllLines("../../../data/LearningSkills_Real_Data_Experiments-Original-Inputs-Quiz-SkillsQuestionsMask.csv");
            return
                csv.Select(line => line.Split(',').Select(val => bool.Parse(val)).ToArray()).ToArray();
        }

        public static bool[][] LoadQuestionsVsUsers()
        {
            var csv = File.ReadAllLines("../../../data/LearningSkills_Real_Data_Experiments-Original-Inputs-IsCorrect.csv");
            var questionsArray =
                csv.Select(line => line.Split(',').Select(val => bool.Parse(val)).ToArray()).ToArray();
            return TransposeRowsAndColumns(questionsArray);
        }

        public static int[][] GetIndexes(bool[][] input)
        {
            var mainList = new List<int[]>();
            foreach (var questionSkills in input)
            {
                var list = new List<int>();
                for (var i = 0; i < questionSkills.Length; i++)
                {
                    if (questionSkills[i])
                    {
                        list.Add(i);
                    }
                }
                var arr = list.ToArray();
                mainList.Add(arr);
            }
            var arrMain = mainList.ToArray();
            return arrMain;
        }

        public static VariableArray<VariableArray<int>,int[][]> ConvertIndexes(int[][] input, Range outer)
        {
            var innerSizesVar = Variable.Array<int>(outer);
            Range inner = new Range(innerSizesVar[outer]);
            var aObs = Variable.Array(Variable.Array<int>(inner), outer);
            var innerSizes = new int[input.Length];
            for (int i = 0; i < input.Length; i++)
                innerSizes[i] = input[i].Length;
            innerSizesVar.ObservedValue = innerSizes;
            aObs.ObservedValue = input;
            return aObs;
        }


        public static void Run()
        {
            var questionVsSkills = LoadQuestionsVsSkills();
            var questiosVsUsers = LoadQuestionsVsUsers();
            var indexes = GetIndexes(questionVsSkills);

            var skills = new Range(7);
            var skillsVars = Variable.Array<bool>(skills).Named("skillsVars");
            skillsVars[skills] = Variable.Bernoulli(0.5).ForEach(skills);

            var questions = new Range(48);
            var correctAnswers = Variable.Array<bool>(questions);

            var indices = ConvertIndexes(indexes, questions);

            using (Variable.ForEach(questions))
            {
                var results = Variable.Subarray<bool>(skillsVars, indices[questions]);
                var hasAllRelevantSkills = Variable.AllTrue(results);
                using (Variable.If(hasAllRelevantSkills))
                    correctAnswers[questions].SetTo(Variable.Bernoulli(0.9));
                using (Variable.IfNot(hasAllRelevantSkills))
                    correctAnswers[questions].SetTo(Variable.Bernoulli(0.2));
            }
            var engine = new InferenceEngine();
            engine.ShowProgress = false;
            foreach (var userAnswers in questiosVsUsers)
            {
                correctAnswers.ObservedValue = userAnswers;
                dynamic result = engine.Infer(skillsVars);
                foreach (var item in result)
                {
                    Console.Write(item.ToString().Substring(9));
                }
                Console.WriteLine();
            }
        }
    }
}
