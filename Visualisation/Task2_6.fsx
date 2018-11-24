#I "../packages/Deedle.1.2.5"
#I "../packages/MathNet.Numerics.3.20.2"
#I "../packages/MathNet.Numerics.FSharp.3.20.2"

#I "../packages"
#r "../packages/XPlot.GoogleCharts.1.5.0/lib/net45/XPlot.GoogleCharts.dll"
#r "../packages/XPlot.Plotly.1.5.0/lib/net45/XPlot.Plotly.dll"


#load "MathNet.Numerics.fsx"

open System
open MathNet.Numerics.Distributions
open XPlot.Plotly

Bernoulli.Samples(0.3)
|> Seq.take 10
|> Seq.toList
|> printfn "%A"

let rec run list =
    match List.length list with
    | 10000 -> list
    | _ ->
        let currProb = ContinuousUniform.Sample(0.0, 1.0)
        let samplesRight = Bernoulli.Samples(currProb)
                            |> Seq.take 10
                            |> Seq.where (fun x -> x = 1)
                            |> Seq.toList
        if samplesRight.Length = 3
        then run (currProb :: list)
        else run list

let results = run []
printf "%A" results

Histogram(x = results)
|> Chart.Plot
|> Chart.WithWidth 700
|> Chart.WithHeight 500
|> Chart.Show