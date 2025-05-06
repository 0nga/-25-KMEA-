package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.AgentProfileType;
import nl.uu.trafficmas.datamodel.DataModelXML;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class GetAgentSpawnInformationTest {

	@Test
	public void test() throws ParserConfigurationException, SAXException, IOException {
		String dir 			= System.getProperty("user.dir")+"/tests/XML/getRoutesAgentTypeSpawnProbabilities/";
		String xmlTest 		= "XMLTest.xml";
		Document agentProfileDoc 	= DataModelXML.loadDocument(dir, xmlTest);
		HashMap<String,LinkedHashMap<AgentProfileType, Double>> routesMap = new HashMap<String, LinkedHashMap<AgentProfileType, Double>>();
		LinkedHashMap<AgentProfileType,Double> agentsAndSpawnProbs = new LinkedHashMap<AgentProfileType, Double>();
		routesMap = DataModelXML.getRoutesAgentTypeSpawnProbabilities(agentProfileDoc);
		agentsAndSpawnProbs = routesMap.get("all");
		assertEquals(agentsAndSpawnProbs.get(AgentProfileType.Normal),0.5,0.0);
		assertEquals(agentsAndSpawnProbs.get(AgentProfileType.OldLady),0.3,0.0);
		assertEquals(agentsAndSpawnProbs.get(AgentProfileType.HotShot),0.2,0.0);
		assertEquals(routesMap.get("route1").get(AgentProfileType.HotShot),0.5,0.0);
	}
}
