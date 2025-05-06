package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.LinkedHashMap;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class XMLSpawnProbabilityTest {

	@Test
	public void xmlSpawnProbability() throws ParserConfigurationException, SAXException, IOException {
		String dir 				= System.getProperty("user.dir")+"/tests/XML/AgentSpawnProbability/";
		String agentProfileXML 	= "AgentProfileTypesTest.xml";
		Document agentProfDoc	= DataModelXML.loadDocument(dir, agentProfileXML);
		LinkedHashMap<String, Double> probs = DataModelXML.getAgentSpawnProbability(agentProfDoc);
		assertEquals(probs.get("all"),0.01,0);
	}

}
