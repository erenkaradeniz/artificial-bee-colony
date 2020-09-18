# -*- coding: utf-8 -*-
# @author: Eren Karadeniz
# ABC Algorithm

using Random

io = open("output.txt", "w");

const NP = 20
const FoodNumber = Int(NP / 2)
const limit = 50
const maxCycle = 2500
const D = 30::Int64
const lb = -5.12
const ub = 5.12
const runtime = 30
const precalc = ub - lb

Foods = zeros(Float64, FoodNumber, D)
f = ones(Float64, FoodNumber)
fitness = ones(Float64, FoodNumber) * typemax(Int64)
trial = zeros(FoodNumber)
prob = zeros(Float64, FoodNumber)
solution = zeros(Float64, D)
GlobalParams = zeros(Float64, D)
GlobalMins = ones(Float64, runtime)
GlobalMin = typemax(Float64)

function CalculateFitness(fun)
    if fun >= 0
        result = 1 / (fun + 1)
    else
        result = 1 + abs(fun)
    end

    return result
end

function MemorizeBestSource()
    global GlobalMin
    global GlobalParams
    for i = 1:FoodNumber
        if f[i] < GlobalMin
            GlobalMin = copy(f[i])
            GlobalParams = copy(Foods[i, :])
        end
    end
end

function init(index::Int64)
    for j = 1:D
        r = rand()
        Foods[index, j] = r * precalc + lb
    end

    solution = copy(Foods[index, :])

    f[index] = calculateFunc(solution)
    fitness[index] = CalculateFitness(f[index])
    trial[index] = 0
end

function initial()
    global GlobalMin
    global GlobalParams
    for i = 1:FoodNumber
        init(i)
    end
    GlobalMin = f[1]
    GlobalParams = copy(Foods[1, :])
end

function SendEmployedBees()
    for i = 1:FoodNumber
        r = rand()
        param2change = floor(Int, r * D) + 1

        r = rand()
        neighbour = floor(Int, r * FoodNumber) + 1
        while (neighbour == i)
            r = rand()
            neighbour = floor(Int, r * FoodNumber) + 1
        end

        solution = copy(Foods[i, :])

        r = rand()
        solution[param2change] =
            Foods[i, param2change] +
            (Foods[i, param2change] - Foods[neighbour, param2change]) *
            (r - 0.5) *
            2

        if solution[param2change] < lb
            solution[param2change] = lb
        end
        if solution[param2change] > ub
            solution[param2change] = ub
        end

        ObjValSol = calculateFunc(solution)
        FitnessSol = CalculateFitness(ObjValSol)

        if FitnessSol > fitness[i]
            trial[i] = 0

            Foods[i, :] = copy(solution)
            f[i] = ObjValSol
            fitness[i] = FitnessSol
        else
            trial[i] = trial[i] + 1
        end
    end
end

function CalculateProbabilities()
    maxfit = copy(maximum(fitness))

    for i = 1:FoodNumber
        prob[i] = (0.9 * (fitness[i] / maxfit)) + 0.1
    end
end

function SendOnlookerBees()
    i = 1
    t = 0

    while (t < FoodNumber)
        r = rand()
        if r < prob[i]
            t = t + 1
            r = rand()
            param2change = floor(Int, r * D) + 1

            r = rand()
            neighbour = floor(Int, r * FoodNumber) + 1

            while (neighbour == i)
                r = rand()
                neighbour = floor(Int, r * FoodNumber) + 1
            end

            solution = copy(Foods[i, :])

            r = rand()
            solution[param2change] =
                Foods[i, param2change] +
                (Foods[i, param2change] - Foods[neighbour, param2change]) *
                (r - 0.5) *
                2

            if solution[param2change] < lb
                solution[param2change] = lb
            end
            if solution[param2change] > ub
                solution[param2change] = ub
            end

            ObjValSol = calculateFunc(solution)
            FitnessSol = CalculateFitness(ObjValSol)

            if FitnessSol > fitness[i]
                trial[i] = 0

                Foods[i, :] = copy(solution)

                f[i] = ObjValSol
                fitness[i] = FitnessSol
            else
                trial[i] = trial[i] + 1
            end
        end

        i = i + 1

        if i == FoodNumber
            i = 1
        end
    end
end

function SendScoutBees()
    global trial

    if maximum(trial) >= limit
        init(argmax(trial))
    end
end

function ABC()
    mean = 0
    out = string("Function: ", calculateFunc, "\n")
    for run = 1:runtime
        initial()
        MemorizeBestSource()
        for iter = 1:maxCycle
            SendEmployedBees()
            CalculateProbabilities()
            SendOnlookerBees()
            MemorizeBestSource()
            SendScoutBees()
        end

        for j = 1:D
            out =
                out * string("GlobalParam[", (j), "]: ", GlobalParams[j], "\n")
        end
        out = out * string(run, ".run ", GlobalMin, "\n")
        GlobalMins[run] = GlobalMin
        mean = mean + GlobalMin
    end
    mean = mean / runtime
    out = out * string("Means of ", runtime, " runs: ", mean, "\n")
    print(out)
    write(io, out)
    close(io)
end

function Griewank(sol)
    top = 0
    top1 = 0
    top2 = 0

    for j = 1:D
        top1 = top1 + sol[j]^2
        top2 = top2 * cos(((sol[j] / sqrt(j + 1)) * pi) / 180)
    end

    top = (1 / 4000) * top1 - top2 + 1
    return top
end


function Rastrigin(sol)
    top = 0
    for j = 1:D
        top = top + (sol[j]^Float64(2) - 10 * cos(2 * pi * sol[j]) + 10)
    end
    return top
end

function Rosenbrock(sol)
    top = 0
    for j = 1:D - 1
        top = top + 100 * ((sol[j + 1] - (sol[j]^2)))^2 + ((sol[j] - 1)^2)
    end
    return top
end

function Sphere(sol)
    top = 0
    for j = 1:D
        top = top + sol[j] * sol[j]
    end
    return top
end

calculateFunc = Griewank

ABC()