package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.io.IOException;
import java.util.HashMap;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.AgentProfileType;
import nl.uu.trafficmas.datamodel.DataModelXML;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class XMLAgentTypeDistributionTest {

	@Test
	public void xmlAgentTypeDistribution() throws ParserConfigurationException, SAXException, IOException {
		String dir 					= System.getProperty("user.dir")+"/tests/XML/GetAgentProfileType/";
		String agentProfileXML 		= "AgentProfileTypesTest.xml";
		Document agentProfileDoc 	= DataModelXML.loadDocument(dir, agentProfileXML);
		
		HashMap<AgentProfileType, Double> dist = DataModelXML.getAgentProfileTypeDistribution(agentProfileDoc);
		
		assertNotNull(dist);
		assertEquals(dist.size(),3);
		assertEquals(dist.get(AgentProfileType.HotShot), 0.05,0);	
		assertEquals(dist.get(AgentProfileType.OldLady), 0.2,0);	
		assertEquals(dist.get(AgentProfileType.Normal), 0.75,0);	
	}
}
