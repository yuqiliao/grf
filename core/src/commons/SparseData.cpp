/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#include <fstream>
#include <sstream>

#include "SparseData.h"
#include "utility.h"

SparseData::SparseData() :
    SparseData(NULL, std::vector<std::string>(), 0, 0) {}

SparseData::SparseData(Eigen::SparseMatrix<double>* data,
                       std::vector<std::string> variable_names,
                       size_t num_rows,
                       size_t num_cols):
    data(data) {
  this->variable_names = variable_names;
  this->num_rows = num_rows;
  this->num_cols = num_cols;
}

SparseData::~SparseData() {
  if (!externalData) {
    delete data;
  }
}

double SparseData::get(size_t row, size_t col) const {
  return data->coeff(row, col);
}

void SparseData::reserve_memory() {
  data = new Eigen::SparseMatrix<double>(num_rows, num_cols);
}

void SparseData::set(size_t col, size_t row, double value, bool& error) {
  data->coeffRef(row, col) = value;
}
