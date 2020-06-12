db.createUser(
  {
      user: "covid_user",
      pwd: "covid_pass",
      roles: [
          {
              role: "readWrite",
              db: "covid_apis_db"
          }
      ]
  }
);
