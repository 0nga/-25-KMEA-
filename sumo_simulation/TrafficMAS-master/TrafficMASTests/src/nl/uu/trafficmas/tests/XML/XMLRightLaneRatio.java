package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.*;

import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class XMLRightLaneRatio {

	@Test
	public void test() throws ParserConfigurationException, SAXException, IOException {
		String dir 				= System.getProperty("user.dir")+"/tests/XML/GetRightLaneRatio/";
		
		String agentProfileXML1	= "XMLTest1.xml";
		String agentProfileXML2	= "XMLTest2.xml";
		
		Document agentProfDoc1	= DataModelXML.loadDocument(dir, agentProfileXML1);
		Document agentProfDoc2	= DataModelXML.loadDocument(dir, agentProfileXML2);
		
		double ratio1 = DataModelXML.getRightLaneRatio(agentProfDoc1);
		assertEquals(.6,ratio1,0.0); 
		
		double ratio2 = DataModelXML.getRightLaneRatio(agentProfDoc2);
		assertEquals(-1,ratio2,0.0);  
	}

}
