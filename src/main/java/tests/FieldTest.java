/* ------------------------------------------------------------------------- */
/*   Copyright (C) 2021 Marius C. Silaghi
                Author: Marius Silaghi: msilaghi@fit.edu
                Florida Tech, Human Decision Support Systems Laboratory
   
       This program is free software; you can redistribute it and/or modify
       it under the terms of the GNU Affero General Public License as published by
       the Free Software Foundation; either the current version of the License, or
       (at your option) any later version.
   
      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.
  
      You should have received a copy of the GNU Affero General Public License
      along with this program; if not, write to the Free Software
      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.              */
/* ------------------------------------------------------------------------- */
package tests;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class FieldTest {

	@Test
	void test() {
		int l = cnn.Field.getIndex(0, 1, 0, 1);
		assertTrue(l == 0, "0 should be 0");
		assertAll("fields addressing",
				() -> assertEquals(0, cnn.Field.getIndex(0, 3, 0, 4)),
				() -> assertEquals(1, cnn.Field.getIndex(1, 3, 0, 4)),
				() -> assertEquals(2, cnn.Field.getIndex(2, 3, 0, 4)),
				() -> assertEquals(3, cnn.Field.getIndex(0, 3, 1, 4)),
				() -> assertEquals(4, cnn.Field.getIndex(1, 3, 1, 4)),
				() -> assertEquals(5, cnn.Field.getIndex(2, 3, 1, 4))				
		);
	}

}
