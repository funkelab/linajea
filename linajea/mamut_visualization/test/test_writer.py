from linajea.mamut_visualization import MamutWriter, MamutReader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestReader(MamutReader):
    def read_data(self, data):
        cell1 = self.create_cell([0, 1, 2, 3], 0, 1)
        cell2 = self.create_cell([1, 2, 3, 4], 0, 2)
        cells = [cell1, cell2]

        edge1 = self.create_edge(2, 1, 0)
        track1 = self.create_track(0, 1, 2, 3, [edge1])
        return cells, [track1]


if __name__ == "__main__":
    writer = MamutWriter()
    writer.add_data(TestReader(), None)
    writer.write("140521_raw.xml", "test_mamut_writer.xml")
