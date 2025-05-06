package nl.uu.trafficmas.datamodel;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.AgentProfileType;
import nl.uu.trafficmas.norm.MergeNormScheme;
import nl.uu.trafficmas.norm.NormScheme;
import nl.uu.trafficmas.norm.SanctionType;
import nl.uu.trafficmas.organisation.CommunicationHub;
import nl.uu.trafficmas.organisation.Organisation;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Lane;
import nl.uu.trafficmas.roadnetwork.LaneType;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;
import nl.uu.trafficmas.roadnetwork.Sensor;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

public class DataModelXML implements DataModel {

	private Document masDoc;
	private Document nodesDoc;
	private Document edgesDoc;
	private Document routesDoc;
	private Document agentProfilesDoc;
	private Document sensorsDoc;
	private Document normsDoc;
	private Document orgsDoc;


	public DataModelXML(String dir, String masXML) throws SAXException, IOException, ParserConfigurationException  {
		setup(dir,masXML);
	}
	
	private void setup(String dir, String masXML) throws SAXException, IOException, ParserConfigurationException {
		masDoc = DataModelXML.loadDocument(dir, masXML);
		
		String nodesXML 			= masDoc.getDocumentElement().getElementsByTagName("nodes").item(0).getAttributes().getNamedItem("value").getTextContent();
		String edgesXML 			= masDoc.getDocumentElement().getElementsByTagName("edges").item(0).getAttributes().getNamedItem("value").getTextContent();
		String routesXML 			= masDoc.getDocumentElement().getElementsByTagName("routes").item(0).getAttributes().getNamedItem("value").getTextContent();
		String agentProfilesXML 	= masDoc.getDocumentElement().getElementsByTagName("agentprofiles").item(0).getAttributes().getNamedItem("value").getTextContent();

		nodesDoc			= DataModelXML.loadDocument(dir, nodesXML);
		edgesDoc			= DataModelXML.loadDocument(dir, edgesXML);
		routesDoc			= DataModelXML.loadDocument(dir, routesXML);
		agentProfilesDoc	= DataModelXML.loadDocument(dir, agentProfilesXML);
		sensorsDoc			= null;
		normsDoc			= null;
		orgsDoc				= null;
		
		NodeList sensorNodes		= masDoc.getDocumentElement().getElementsByTagName("sensors");
		if(sensorNodes.getLength()>0){
			String sensorXML		= sensorNodes.item(0).getAttributes().getNamedItem("value").getTextContent();
			sensorsDoc				= DataModelXML.loadDocument(dir, sensorXML);
		}
		
		NodeList normNodes = masDoc.getDocumentElement().getElementsByTagName("norms");
		if(normNodes.getLength()>0){
			String normsXML			= normNodes.item(0).getAttributes().getNamedItem("value").getTextContent();
			normsDoc				= DataModelXML.loadDocument(dir, normsXML);
		}
		
		NodeList orgNodes = masDoc.getDocumentElement().getElementsByTagName("organisations");
		if(normNodes.getLength()>0){
			String normsXML		= masDoc.getDocumentElement().getElementsByTagName("organisations").item(0).getAttributes().getNamedItem("value").getTextContent();
			orgsDoc				= DataModelXML.loadDocument(dir, normsXML);
		}
	}

	public static Document loadDocument(String dir, String xml)
			throws ParserConfigurationException, SAXException, IOException {
		File fXmlFile 						= new File(dir+xml);
		DocumentBuilderFactory dbFactory 	= DocumentBuilderFactory.newInstance();
		DocumentBuilder dBuilder 			= dbFactory.newDocumentBuilder();
		Document doc 						= dBuilder.parse(fXmlFile);
		doc.getDocumentElement().normalize();
		return doc;
	}

	@Override
	public RoadNetwork instantiateRoadNetwork() {
		if(sensorsDoc == null){
			return instantiateRoadNetwork(this.nodesDoc, this.edgesDoc);
		} else {
			return instantiateRoadNetwork(this.nodesDoc, this.edgesDoc, this.sensorsDoc);
		}
	}
	
	/**
	 * Creates a RoadNetwork in the MAS according to 'nodesXML' and 'edgesXML'.
	 * The RoadNetwork object is also validated by validateRoadNetwork(). 
	 * These xml files must be present in the directory given by the variable 'dir'.
	 * @param dir
	 * @param nodesXML
	 * @param edgesXML
	 * @return a RoadNetwork, if the RoadNetwork is not correctly validated, this method will return null.
	 * @throws ParserConfigurationException 
	 * @throws IOException 
	 * @throws SAXException 
	 */
	public static RoadNetwork instantiateRoadNetwork(Document nodeDoc, Document edgeDoc) {
		HashMap<String,Node> nodes 	= DataModelXML.extractNodes(nodeDoc);
		ArrayList<Node> nodeList 	= new ArrayList<Node>(nodes.values());
		ArrayList<Edge> edges 		= extractEdges(edgeDoc,nodes);
		
		RoadNetwork rn = new RoadNetwork(nodeList, edges);
		
		if(rn.validateRoadNetwork()){
			return rn;
		} else{
			return null;
		}
	}
	
	public static RoadNetwork instantiateRoadNetwork(Document nodeDoc, Document edgeDoc, Document sensorDoc) {
		HashMap<String,Node> nodes 	= DataModelXML.extractNodes(nodeDoc);
		ArrayList<Node> nodeList 	= new ArrayList<Node>(nodes.values());
		ArrayList<Edge> edges 		= extractEdges(edgeDoc,nodes);
		
		RoadNetwork rn = new RoadNetwork(nodeList, edges);
		
		// Add sensors to the road network.
		HashMap<String,Sensor> sensorMap 		= getSensors(rn, sensorDoc);
		rn.addSensors(sensorMap);		
		
		if(rn.validateRoadNetwork()){
			return rn;
		} else{
			return null;
		}
	}
		
	/**
	 * Extracts all values needed from 'nodesXML' and returns a HashMap filled with Node objects.
	 * The xml file must be present in the directory given by the variable 'dir'.
	 * @param dir
	 * @param nodesXML
	 * @return a map with all Node objects.
	 * @throws ParserConfigurationException 
	 * @throws IOException 
	 * @throws SAXException 
	 */
	public static HashMap<String,Node> extractNodes(Document nodeDoc) {
		NodeList nodeList = nodeDoc.getDocumentElement().getElementsByTagName("node");
		
		HashMap<String,Node>nodes = new LinkedHashMap<String,Node>();
		for (int i = 0; i < nodeList.getLength(); i++) {
			org.w3c.dom.Node n 		= nodeList.item(i);
			NamedNodeMap attributes = n.getAttributes();
			String id 				= attributes.getNamedItem("id").getTextContent();
			String x 				= attributes.getNamedItem("x").getTextContent();
			String y 				= attributes.getNamedItem("y").getTextContent();
			
			Node node = new Node(id,Double.parseDouble(x),Double.parseDouble(y));
			nodes.put(id,node);
		}
		
		return nodes;
	}
	
	/**
	 * Extracts all values needed from 'edgesXML' and returns an ArrayList filled with Edge objects. 
	 * The 'nodes' is used, since Edge object need to be connected to Node objects.
	 * The XML file must be present in the directory given by the variable 'dir'.
	 * @param dir
	 * @param edgesXML
	 * @param nodes 
	 * @return a List containing all Edge objects.
	 * @throws IOException 
	 * @throws SAXException 
	 * @throws ParserConfigurationException 
	 */
	public static ArrayList<Edge> extractEdges(Document edgeDoc, HashMap<String,Node> nodes) {
		NodeList edgeList= edgeDoc.getElementsByTagName("edge");
		ArrayList<Edge> edges = new ArrayList<Edge>();

		for (int i = 0; i < edgeList.getLength(); i++) {
			org.w3c.dom.Node n 	= edgeList.item(i);
			NamedNodeMap attr 	= n.getAttributes();
			extractEdge(edges, attr,nodes);
		}
		
		return edges;
	}

	/**
	 * Extracts the values from 'edgeAttributes' 
	 * and uses the appropriate Node objects from 'nodes' to add an Edge object to the list 'edges'.
	 * @param edges
	 * @param attr
	 * @param nodes
	 */
	public static void extractEdge(ArrayList<Edge> edges,
			NamedNodeMap attr,
			HashMap<String,Node> nodes) {
		String from 		= attr.getNamedItem("from").getTextContent();
		String to 			= attr.getNamedItem("to").getTextContent();
		String id 			= attr.getNamedItem("id").getTextContent();
		String numberLanes 	= attr.getNamedItem("numLanes").getTextContent();
		String priority		= attr.getNamedItem("priority").getTextContent();		
		
		ArrayList<Lane> lanes = new ArrayList<Lane>();
		int numberLanesInt = Integer.parseInt(numberLanes);

		for (int i = 0; i < numberLanesInt; i++) {
			lanes.add(new Lane(LaneType.Normal,(byte) i)); //TODO: find some way to encode lanetype in xml
			lanes.get(i).setID(id+"_"+i); //TODO: write a test to check this 
		}
		
		for (int i = 0; i < numberLanesInt; i++) {
			if(i!= numberLanesInt-1) {
				lanes.get(i).setLeftLane(lanes.get(i+1));
			}
		}
		
		Node fromNode 	= nodes.get(from);
		Node toNode 	= nodes.get(to);
		
		double distance = Node.nodeDistance(fromNode, toNode);
		
		Road road = new Road(id, distance, lanes, Integer.parseInt(priority));
		
		Edge n = new Edge(fromNode, toNode, road);
		edges.add(n);
	}
	
	@Override
	public HashMap<String,LinkedHashMap<AgentProfileType, Double>> getRoutesAgentTypeSpawnProbabilities(){
		return getRoutesAgentTypeSpawnProbabilities(this.agentProfilesDoc);
	}
	
	/**
	 * TODO: Document
	 * @param agentProfileXML
	 * @return
	 */
	public static HashMap<String,LinkedHashMap<AgentProfileType, Double>> getRoutesAgentTypeSpawnProbabilities(Document agentProfileDoc){
		HashMap<String,LinkedHashMap<AgentProfileType, Double>> routesMap = new HashMap<String, LinkedHashMap<AgentProfileType, Double>>();
		NodeList routeList= agentProfileDoc.getElementsByTagName("route");
		for(int i=0;i<routeList.getLength(); i++){
			org.w3c.dom.Node n 		= routeList.item(i);
			NamedNodeMap attributes = n.getAttributes();
			String id 				= attributes.getNamedItem("id").getTextContent();
			LinkedHashMap<AgentProfileType, Double> agentsAndDist = new LinkedHashMap<AgentProfileType, Double>();
			routesMap.put(id, agentsAndDist);
			for (int j=0; j< attributes.getLength(); j++){
				if(!attributes.item(j).getNodeName().equals("id") && !attributes.item(j).getNodeName().equals("spawn-probability")){
					AgentProfileType agent = AgentProfileType.getAgentProfileType(attributes.item(j).getNodeName());
					double spawnProb = Double.valueOf(attributes.item(j).getNodeValue());
					agentsAndDist.put(agent, spawnProb);
				}
			}
		}
		
		return routesMap;
	}
	
	@Override
	public boolean getMultipleRoutesValue() {
		return getMultipleRoutesValue(this.agentProfilesDoc);
	}
	
	public static boolean getMultipleRoutesValue(Document agentProfileDoc) {
		NamedNodeMap attributes = agentProfileDoc.getElementsByTagName("agents").item(0).getAttributes();
		String spawnProbString = attributes.getNamedItem("multiple-routes").getTextContent();
		return Boolean.parseBoolean(spawnProbString);
	}
	
	@Override
	public double getRightLaneRatio(){
		return getRightLaneRatio(this.agentProfilesDoc);
	}
	
	public static double getRightLaneRatio(Document agentProfileDoc){
		NamedNodeMap attributes = agentProfileDoc.getElementsByTagName("agents").item(0).getAttributes();
		String rightLaneRatioString = "-1";
		if(attributes.getNamedItem("right-lane-ratio") != null){
			rightLaneRatioString =attributes.getNamedItem("right-lane-ratio").getTextContent();
		}
		return Double.parseDouble(rightLaneRatioString);
	}
	
	@Override
	public LinkedHashMap<String, Double> getAgentSpawnProbability() {
		return getAgentSpawnProbability(this.agentProfilesDoc);
	}
	
	/**
	 * Returns the probability for an agent to spawn on each tick,
	 * this value is extracted from 'agentProfilesXML'.
	 * The xml file must be present in the directory given by the variable 'dir'.
	 * @param dir
	 * @param agentProfilesXML
	 * @return a value between 0 and 1, including 0 and 1.
	 * @throws ParserConfigurationException 
	 * @throws IOException 
	 * @throws SAXException 
	 */
	public static LinkedHashMap<String, Double> getAgentSpawnProbability(Document agentProfileDoc) {
		LinkedHashMap<String, Double> agentSpawnProbabilities = new LinkedHashMap<String, Double>();
		NodeList routeList= agentProfileDoc.getElementsByTagName("route");
		for(int i=0;i<routeList.getLength(); i++){
			org.w3c.dom.Node n 		= routeList.item(i);
			NamedNodeMap attributes = n.getAttributes();
			String id 				= attributes.getNamedItem("id").getTextContent();
			String spawnProb		= attributes.getNamedItem("spawn-probability").getTextContent();
			agentSpawnProbabilities.put(id, Double.parseDouble(spawnProb));
		}		
		return agentSpawnProbabilities;
	}
	
	@Override
	public LinkedHashMap<AgentProfileType, Double> getAgentProfileTypeDistribution() {
		return getAgentProfileTypeDistribution(this.agentProfilesDoc);
	}
	
	/**
 	 * Extracts the agentProfileTypeDistribution from 'agentProfilesXML' and returns it in a HashMap.
	 * The XML file must be present in the directory given by the variable 'dir'.
	 * @param dir
	 * @param agentProfilesXML
	 * @return a map containing the AgentProfile and the chance of it occurring in a value between 0 and 1.
	 * @throws IOException 
	 * @throws SAXException 
	 * @throws ParserConfigurationException 
	 */
	public static LinkedHashMap<AgentProfileType, Double> getAgentProfileTypeDistribution(Document agentProfileDoc) {
		NodeList agentList = agentProfileDoc.getElementsByTagName("agent");
		LinkedHashMap<AgentProfileType, Double> agentTypeAndDist = new LinkedHashMap<AgentProfileType, Double>();
		
		for (int i = 0; i < agentList.getLength(); i++) {
			org.w3c.dom.Node n = agentList.item(i);
			NamedNodeMap attr = n.getAttributes();
			String type = attr.getNamedItem("role").getTextContent();
			String dist = attr.getNamedItem("dist").getTextContent();

			AgentProfileType agentType = AgentProfileType.valueOf(type);
			double distVal = Double.parseDouble(dist);
			agentTypeAndDist.put(agentType, distVal);
		}
		return agentTypeAndDist;
	}

	@Override
	public int getSimulationLength() {
		return simulationLength(this.masDoc);
	}
	
	/**
	 * Extracts the simulation length from 'masXML' and converts it to an Integer value.
	 * The XML file must be present in the directory given by the variable 'dir'.
	 * @param dir
	 * @param masXML
	 * @return an Integer with the time in seconds concerning how long SUMO will run.
	 * @throws ParserConfigurationException 
	 * @throws IOException 
	 * @throws SAXException 
	 */
	public static int simulationLength(Document masDoc) {
		NamedNodeMap attributes = masDoc.getElementsByTagName("mas").item(0).getAttributes();
		String simulationLengthString = attributes.getNamedItem("simulationlength").getTextContent();

		return Integer.parseInt(simulationLengthString);
	}
	
	@Override
	public ArrayList<Route> getRoutes(RoadNetwork rn) {
		return getRoutes(rn,this.routesDoc);
	}
	
	/**
	 * Extracts a list of Route objects, by reading 'routesXML'.
	 * The xml file must be present in the directory given by the variable 'dir'.
	 * @param rn
	 * @param dir
	 * @param routesXML
	 * @return an ArrayList of Route objects
	 * @throws ParserConfigurationException 
	 * @throws IOException 
	 * @throws SAXException 
	 */
	public static ArrayList<Route> getRoutes(RoadNetwork rn, Document routesDoc) {
		NodeList routeList = routesDoc.getElementsByTagName("route");
		ArrayList<Route> routes = new ArrayList<Route>();

		for (int i = 0; i < routeList.getLength(); i++) {
			org.w3c.dom.Node n = routeList.item(i);
			NamedNodeMap attr = n.getAttributes();
			String id= attr.getNamedItem("id").getTextContent();
			String edgesString= attr.getNamedItem("edges").getTextContent();
			ArrayList<Edge> edges = new ArrayList<Edge>();

			String[] edgesSplitted = edgesString.split(" ");
			for(String roadID : edgesSplitted) {
				edges.add(rn.getEdge(roadID));
			}
			
			Route route = new Route(id,edges);
			routes.add(route);
		}
		
		return routes;
	}
	
	

	/**
	 * 
	 * @param rn
	 * @param sensorsDoc
	 * @return
	 */
	public static HashMap<String, Sensor> getSensors(RoadNetwork rn, Document sensorsDoc){
		HashMap<String,Sensor> sensors = new HashMap<String, Sensor>();
		NodeList sensorList = sensorsDoc.getElementsByTagName("laneAreaDetector");
		
		for (int i = 0; i < sensorList.getLength(); i++) {
			org.w3c.dom.Node n = sensorList.item(i);
			NamedNodeMap attr = n.getAttributes();
			String id 			= attr.getNamedItem("id").getTextContent();
			String lane 		= attr.getNamedItem("lane").getTextContent();
			String position 	= attr.getNamedItem("pos").getTextContent();
			String length 		= attr.getNamedItem("length").getTextContent();
			String frequency 	= attr.getNamedItem("freq").getTextContent();
			
			String roadID 		= lane.substring(0, lane.length()-2);
			String laneIndex 	= lane.substring(lane.length()-1, lane.length());
			Road r = rn.getRoadFromID(roadID);
			Lane l = r.laneList.get((Integer.parseInt(laneIndex)));
			Sensor sensor = new Sensor(id, l, Double.parseDouble(position), Double.parseDouble(length), Integer.parseInt(frequency));
			sensors.put(id, sensor);
		}
		return sensors;
	}
	
	public static Map<String, NormScheme> getNormSchemes(Map<String,Sensor> sensors, Document normsDoc) {
		Map<String,NormScheme> normSchemes = new HashMap<String,NormScheme>();
		NodeList normSchemeList = normsDoc.getElementsByTagName("norm_scheme");
		
		for (int i = 0; i < normSchemeList.getLength(); i++) {
			List<Sensor> normSensors = new ArrayList<Sensor>();
			org.w3c.dom.Node n = normSchemeList.item(i);
			if (n.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE) {
				Element element = (Element) n;
				String id 			= element.getAttribute("id");
				String sanctionStr	= element.getAttribute("sanction");	
				String packageName	= element.getAttribute("packagename");
				String className	= element.getAttribute("classname");

				NodeList sensorList = element.getElementsByTagName("sensor");
				for (int j = 0; j < sensorList.getLength(); j++) {
					String sensID = sensorList.item(j).getAttributes().getNamedItem("id").getTextContent();
					normSensors.add(sensors.get(sensID));
				}
				
				NodeList childNodes = n.getChildNodes();
				Map<String,String> attributes = new HashMap<String,String>();
				for (int j = 0; j < childNodes.getLength(); j++) {
					org.w3c.dom.Node childNode = childNodes.item(j);
					if(childNode.getNodeName().equals("sensor") || childNode.getNodeName().equals("#text")) {
						continue;
					}
					attributes.put(childNode.getNodeName(),childNode.getTextContent());
				}
				
				SanctionType sanctionType = SanctionType.valueOf(sanctionStr);
				
				//TODO: Apply this to different norm schemes using "classname" param in xml.
				NormScheme normScheme = null;
				normScheme = strToNormScheme(normSensors, id, packageName,
						className, sanctionType);
			
				if(normScheme == null) {
					normScheme = new MergeNormScheme(id,sanctionType,normSensors);
				}
				normScheme.addAttributes(attributes);
				normSchemes.put(id, normScheme);
			}	
		}		
		return normSchemes;
	}

	public static NormScheme strToNormScheme(List<Sensor> normSensors,
			String id, String packageName, String className,
			SanctionType sanctionType)  {
		NormScheme normScheme = null;
		try {
			packageName = (packageName.isEmpty()) ? ("nl.uu.trafficmas.norm") : packageName;
			Class<?> cls = Class.forName(packageName+"."+className);
			normScheme = (NormScheme) cls.getDeclaredConstructor(String.class, SanctionType.class, List.class).newInstance(id,sanctionType,normSensors);
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NoSuchMethodException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InstantiationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return normScheme;
	}

	public Map<String, Organisation> getOrganisations(Map<String,Sensor> sensorMap){
		return getOrganisations(sensorMap, this.normsDoc, this.orgsDoc);
	}
	
	public static Map<String, Organisation> getOrganisations(Map<String,Sensor> sensorMap, Document normDoc, Document orgDoc) {
		Map<String,Organisation> orgMap = new HashMap<String,Organisation>();

		ArrayList<NormScheme> normSchemes = new ArrayList<NormScheme>();
		ArrayList<Sensor> sensors = new ArrayList<Sensor>();
		NodeList organisationList = orgDoc.getElementsByTagName("organisation");
		Map<String, NormScheme> normSchemeMap = getNormSchemes(sensorMap, normDoc);
		
		
		for (int i = 0; i < organisationList.getLength(); i++) {
			 
			org.w3c.dom.Node nNode = organisationList.item(i);
			if (nNode.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE) {
				Element element = (Element) nNode;
				String orgId = element.getAttribute("id");

				NodeList sensorList = element.getElementsByTagName("sensor");
				for (int j = 0; j < sensorList.getLength(); j++) {
					String sensId = sensorList.item(j).getAttributes().getNamedItem("id").getTextContent();
					Sensor sensor = sensorMap.get(sensId);
					sensors.add(sensor);
				}	
				
				NodeList normsSensorList = element.getElementsByTagName("norm");
				for (int k = 0; k < normsSensorList.getLength(); k++) {
					org.w3c.dom.Node nodeNormSensor = normsSensorList.item(k);
					
					String normId = nodeNormSensor.getAttributes().getNamedItem("id").getTextContent();
					NormScheme normScheme = normSchemeMap.get(normId);
					normSchemes.add(normScheme);
				}	 
				Organisation organisation = new Organisation(orgId,normSchemes, sensors);
				orgMap.put(orgId, organisation);
			}
		}
		return orgMap;
	}
	
	public CommunicationHub<Organisation> getCommunicationHub(Map<String,Organisation> orgMap){
		return getCommunicationHub(orgMap, this.orgsDoc);
	}
	
	public static CommunicationHub<Organisation> getCommunicationHub(
			Map<String, Organisation> orgMap, Document orgsDoc) {
		NodeList organisationList = orgsDoc.getElementsByTagName("organisation");
		
		CommunicationHub<Organisation> ch = new CommunicationHub<Organisation>();
		
		for (int i = 0; i < organisationList.getLength(); i++) {
			 
			org.w3c.dom.Node nNode = organisationList.item(i);
			if (nNode.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE) {
				Element element = (Element) nNode;
				String orgId = element.getAttribute("id");

				NodeList normsSensorList = element.getElementsByTagName("observed");
				for (int j = 0; j < normsSensorList.getLength(); j++) {
					String observedid = normsSensorList.item(j).getAttributes().getNamedItem("id").getTextContent();
					ch.addSubscription(orgMap.get(observedid), orgMap.get(orgId));
				}	
			}
		}
		
		
		return ch;
	}

	public static Map<String, Organisation> instantiateOrganisations(
			Document nodeDoc, Document edgeDoc, Document sensorDoc, Document normDoc, Document orgDoc) {
		
		Map<String,Organisation> orgMap = new HashMap<String,Organisation>();
		HashMap<String,Node> nodes 	= DataModelXML.extractNodes(nodeDoc);
		ArrayList<Node> nodeList 	= new ArrayList<Node>(nodes.values());
		ArrayList<Edge> edges 		= extractEdges(edgeDoc,nodes);
		RoadNetwork rn = new RoadNetwork(nodeList, edges);
		
		HashMap<String,Sensor> sensorMap 		= getSensors(rn, sensorDoc);
		Map<String,NormScheme> normSchemeMap	= getNormSchemes(sensorMap, normDoc);
		
		
		NodeList organisationList = orgDoc.getElementsByTagName("organisation");
		
		for (int i = 0; i < organisationList.getLength(); i++) {
			 
			org.w3c.dom.Node nNode = organisationList.item(i);
			if (nNode.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE) {
				Element element = (Element) nNode;
				String orgId = element.getAttribute("id");
				ArrayList<NormScheme> normSchemes = new ArrayList<NormScheme>();
				ArrayList<Sensor> sensors = new ArrayList<Sensor>();

				NodeList sensorList = element.getElementsByTagName("sensor");
				for (int j = 0; j < sensorList.getLength(); j++) {
					String sensId = sensorList.item(j).getAttributes().getNamedItem("id").getTextContent();
					Sensor sensor = sensorMap.get(sensId);
					sensors.add(sensor);
				}
				
				NodeList normsSensorList = element.getElementsByTagName("norm");
				for (int k = 0; k < normsSensorList.getLength(); k++) {
					String normId = normsSensorList.item(k).getAttributes().getNamedItem("id").getTextContent();
					NormScheme normScheme = normSchemeMap.get(normId);
					normSchemes.add(normScheme);
				}	 
				Organisation organisation = new Organisation(orgId,normSchemes, sensors);
				orgMap.put(orgId, organisation);
			}
		}
		return orgMap;
	}

	@Override
	public MASData getMASData() {
		return getMASData(this.masDoc, this.agentProfilesDoc, this.nodesDoc, this.edgesDoc, this.sensorsDoc, this.normsDoc, this.orgsDoc, this.routesDoc);
	}
	
	/**
	 * Returns the MASData data structure which contains data read from several XML files.
	 * MASData contains simulationLength, sumoConfigPath, spawnProbability and agentProfileTypeDistribution.
	 * @param dir
	 * @param masXML
	 * @param sumoConfigXML
	 * @param agentProfilesXML
	 * @return the MASData data structure
	 * @throws IOException 
	 * @throws SAXException 
	 * @throws ParserConfigurationException 
	 */
	public static MASData getMASData(Document masDoc, Document agentProfilesDoc, Document nodeDoc, Document edgeDoc, Document sensorDoc, Document normDoc, Document orgsDoc,Document routesDoc) {
		int simulationLength 								= DataModelXML.simulationLength(masDoc);
		boolean multipleRoutes 								= DataModelXML.getMultipleRoutesValue(agentProfilesDoc);
		LinkedHashMap<String, Double> spawnProbabilities 	= DataModelXML.getAgentSpawnProbability(agentProfilesDoc);
		double rightLaneRatio								= DataModelXML.getRightLaneRatio(agentProfilesDoc);
		HashMap<String, LinkedHashMap<AgentProfileType,Double>> routeAgentTypeSpawnProbabilityMap 	= DataModelXML.getRoutesAgentTypeSpawnProbabilities(agentProfilesDoc);
		Map<String, Organisation> instantiatedOrganisations = null;
		CommunicationHub ch = null;
		if(!(sensorDoc == null || normDoc == null || orgsDoc == null)) {
			instantiatedOrganisations = DataModelXML.instantiateOrganisations(nodeDoc, edgeDoc, sensorDoc, normDoc, orgsDoc);
			ch = DataModelXML.getCommunicationHub(instantiatedOrganisations, orgsDoc);
		}
		return new MASData(simulationLength, spawnProbabilities, rightLaneRatio, multipleRoutes, routeAgentTypeSpawnProbabilityMap,instantiatedOrganisations,ch);
	}

	@Override
	public void close() {
		// TODO Auto-generated method stub
	}
}
