package nl.uu.trafficmas.roadnetwork;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class RoadNetwork {
	private Set<Node> 	nodes;
	private Set<Edge> 	edges;
	private Set<Sensor> sensors;
	private Map<Node,Map<Node,Edge>> nodesEdgeLinks;
	public RoadNetwork() {
		nodes = new HashSet<Node>();
		edges = new HashSet<Edge>();
		sensors = new HashSet<Sensor>();
		nodesEdgeLinks = new HashMap<Node, Map<Node,Edge>>();
	}
	
	public RoadNetwork(ArrayList<Node> nodeList, ArrayList<Edge> edgeList) {
		this();
		addNodes(nodeList);
		addEdges(edgeList);
	}
	
	public Node[] getNodes() {
		Node[] list = new Node[nodes.size()];
		return nodes.toArray(list);
	}
	
	//TODO: use hashmaps so this is faster
	public Edge getEdge(String roadID) {
		
		for(Edge edge : edges) {
			if(edge.getRoad().id.equals(roadID))
				return edge;
		}
		return null;
	}
	public Edge[] getEdges() {
		Edge[] list = new Edge[edges.size()];
		return edges.toArray(list);
	}
	
	public void addNode(Node node) {
		if(!nodes.contains(node)) {
			nodes.add(node);
			nodesEdgeLinks.put(node, new HashMap<Node, Edge>());
		}
	}
	
	public void addNodes(ArrayList<Node> newNodes){
		for(int i = 0; i < newNodes.size(); i++){
			this.addNode(newNodes.get(i));
		}
	}
	
	public void addSensors(HashMap<String,Sensor> sensors){
		for(Entry<String,Sensor> entry : sensors.entrySet()){
			this.sensors.add(entry.getValue());
		}
	}

	public Map<String, Sensor> getSensorMap() {
		Map<String, Sensor> sensorMap = new HashMap<String, Sensor>();
		for(Sensor s : this.sensors){
			sensorMap.put(s.id, s);
		}
		return sensorMap;
	}
	
	public void removeNode(Node node) {
		nodes.remove(node);
	}
	
	public void addEdge(Edge edge) {
		Node from = edge.getFromNode();
		Node to = edge.getToNode();
		Road road = edge.getRoad();
		this.addEdge(from,to,road);
	}
	
	public void addEdge(Node from, Node to, Road road) {
		Edge edge = new Edge(from,to,road);
		if(!edges.contains(edge)) {
			edges.add(edge);
			addNode(from);
			addNode(to);
			nodesEdgeLinks.get(from).put(to, edge);
		}
	}
	
	public void addEdges(ArrayList<Edge> newEdges){
		for(int i = 0; i < newEdges.size(); i++){
			addEdge(newEdges.get(i));
		}
	}
	
	public void removeEdge(Edge edge){
		edges.remove(edge);
	}
	
	public Road getRoadFromID(String roadID){
		for(Edge edge: edges){
			if (edge.getRoad().id.equals(roadID)){
				return edge.getRoad();
			}
		}
		return null;
	}
	
	public boolean validateRoadNetwork() {
		ArrayList<Road> roads = new ArrayList<Road>();
		
		// Check for duplicate nodes
		Set<Node> setNodes = new HashSet<Node>(nodes);
		if (setNodes.size() < nodes.size()){
			return false;
		}
		
		// Check for duplicate edges
		Set<Edge> setEdges = new HashSet<Edge>(edges);
		if (setEdges.size() < edges.size()){
			return false;
		}
		
		// Check for all edges if,
		for (Edge edge : edges) {
			// "from" and "to" are not the same node.
			if(edge.getFromNode().equals(edge.getToNode())){
				return false;
			}
			// the "from" and to "nodes" are present in the RoadNetwork nodes list.
			if(!nodes.contains(edge.getFromNode()) && !nodes.contains(edge.getToNode())){
				return false;
			}
			// every edge has an unique road.
			roads.add(edge.getRoad());
		}
		
		// Check for duplicate Roads.
		Set<Road> setRoads = new HashSet<Road>(roads);
		if (setRoads.size() < roads.size()){
			return false;
		}
		
		// If none of the conditions are met, the RoadNetwork is valid.
		return true;
	}
	
	public void clear(){
		edges.clear();
		nodes.clear();
	}

	public Set<Node> getSuccessors(Node q) {
		return RoadNetwork.getSuccessors(q,edges);
	}
	
	public static Set<Node> getSuccessors(Node q, Set<Edge> edges) {
		Set<Node> successors = new HashSet<Node>();
		for(Edge edge: edges) {
			if(edge.getFromNode().equals(q)) {
				successors.add(edge.getToNode());
			}
		}
		
		return successors;
	}

	public static Edge createSimpleEdge(RoadNetwork rn, Node from, Node to1, String roadName) {
		Lane l = new Lane(LaneType.Normal, (byte) 0);
		List<Lane> laneList = new ArrayList<Lane>();
		laneList.add(l);
		
		Road r = new Road(roadName, Node.nodeDistance(from, to1), laneList, 0);
		Edge edge = new Edge(from, to1, r);	
		rn.addEdge(edge);
		return edge;
	}

	public Edge getEdge(Node q, Node successor) {
		return nodesEdgeLinks.get(q).get(successor);
	}

	public Map<String, Double> getAverageTravelTime() {
		Map<String,Double> averageTravelTime = new HashMap<String, Double>();
		for(Edge edge : edges) {
			averageTravelTime.put(edge.getID(), edge.getRoad().getMeanTravelTime());
		}
		return averageTravelTime;
	}

	public Map<String, Double> getRoadBestLaneAverageTravelTime() {
		Map<String,Double> roadBestLaneAverageTravelTime = new HashMap<String, Double>();
		for(Edge edge : edges) {
			double lowestTravelTime = Double.MAX_VALUE;
			for(Lane lane : edge.getRoad().getLanes()) {
				if(lane.getMeanTravelTime() < lowestTravelTime) {
					lowestTravelTime = lane.getMeanTravelTime(); 
				}
			}
			roadBestLaneAverageTravelTime.put(edge.getID(), lowestTravelTime);
		}
		return roadBestLaneAverageTravelTime;
	}
}
