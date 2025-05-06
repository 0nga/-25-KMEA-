package nl.uu.trafficmas.datamodel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import nl.uu.trafficmas.agent.AgentProfileType;
import nl.uu.trafficmas.organisation.Organisation;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;
import nl.uu.trafficmas.roadnetwork.Sensor;

public interface DataModel {
	/**
	 * Instantiates the RoadNetwork according to the XML files read by the Datamodel.
	 * @return a RoadNetwork, if the RoadNetwork is not correctly validated, this method will return null.
	 */
	public RoadNetwork 	instantiateRoadNetwork();
	
	/**
	 * Returns the MASData, which includes simulationLength, sumoConfigPath, spawnProbability and AgentProfileTypeDistribution.
	 * @return the MASData data structure
	 */
	public MASData 	getMASData();
	
	
	/**
	 * Returns the probability of an agent spawning each tick.
	 * @return a value between 0 and 1, including 0 and 1.
	 */
	public LinkedHashMap<String, Double> getAgentSpawnProbability();
	
	/**
	 * Extracts the simulation length from the MAS.XML file.
	 * @return an Integer with the time in seconds concerning how long SUMO will run.
	 */
	public int getSimulationLength();
	
	/**
	 * Extracts the agentProfileTypeDistribution and returns it in a HashMap.
	 * @return a map containing the AgentProfile and the chance of it occurring in a value between 0 and 1.
	 */
	public HashMap<AgentProfileType, Double> getAgentProfileTypeDistribution();
	
	/**
	 * Returns a list of all routes that can be used by the agents. These routes are stated in the XML files read by the Datamodel.
	 * @param rn
	 * @return an ArrayList of all routes read from the XML file. 
	 */
	public ArrayList<Route> getRoutes(RoadNetwork rn);
	
	/**
	 * 
	 * @param sensorMap
	 * @return
	 */
	public Map<String, Organisation> getOrganisations(Map<String,Sensor> sensorMap);

	
	/**
	 * Not yet implemented
	 */
	public void close();
	
	/**
	 * Returns a hashmap which contains the AgentProfile distribution map on every route as parsed by the route.xml file.
	 * @return
	 */
	public HashMap<String, LinkedHashMap<AgentProfileType, Double>> getRoutesAgentTypeSpawnProbabilities();
	
	/**
	 * Returns a boolean value which indicates if agents have individual spawn rates for each route.
	 * @return true if agents have individual spawn probabilities on each route.
	 */
	public boolean getMultipleRoutesValue();
	
	/**
	 * Returns a double value which indicates the amount of agents that will spawn on the rightmost lane.
	 * @return "-1" if no such value is found in the XML file.
	 */
	public double getRightLaneRatio();
	
}
