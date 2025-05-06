package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class getMultipleRoutesValueTest {

	@Test
	public void test() throws ParserConfigurationException, SAXException, IOException {
		String dir 				= System.getProperty("user.dir")+"/tests/XML/GetMultipleRoutesValue/";
		String agentProfileXML 	= "XMLTest.xml";
		Document agentProfDoc	= DataModelXML.loadDocument(dir, agentProfileXML);
		boolean prob = DataModelXML.getMultipleRoutesValue(agentProfDoc);
		assertEquals(true,prob);
	}
}
