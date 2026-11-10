# Model routing

Jev: given the state, what structured decision should the application use next?

Here's the flow:
1. We start with an input query text.
2. Jev classifies the query type. The "choice" option.
3. Routes to the right model to deal with the user's request.
- Pareto frontier can be used to determine accuracy-latency-cost frontier of the models. Characterize the task space, benchmark the models offline; this would keep changing as the models get better and the plot updates.
- In the end, we can show cost (latency?) reductions using our routing capability as opposed to always running an expensive VLM.

We could use VLMs/MLLMs to design the benchmark.
What would the query classes look like?
Could MapEvolve subsume Polaris (https://github.com/kpatherya/polaris)? Because the latter works with a particular query type that we use FastVLM with.